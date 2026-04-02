"""
TELE 523 – Group 1 · Digital Telemetry Lead Script
====================================================
Produces two primary output files:
  1. monitoring_stream.log   – timestamped stream log for the Monitoring Lead
  2. digital_telemetry.json  – full digital-telemetry analysis for the report

Plus diagnostic plots in results/digital_telemetry/:
  - BER curves (all stations × all schemes)
  - SQNR vs bit-depth comparison bars
  - NRZ vs Manchester eye diagrams (one per station)

Usage
-----
    python digital_telemetry.py [--data-dir PATH] [--out-dir PATH]

Default data-dir : ./modulation_output/raw/
Default out-dir  : ./results/digital_telemetry/
"""

import os
import json
import argparse
import warnings
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
STATIONS      = ["EC_1", "HBW_1", "MM_1", "OV_1", "SM_1", "VGR_1", "WT_1"]
SCHEMES       = ["AM", "FM", "ASK", "FSK", "PSK"]
BIT_DEPTHS    = [8, 10, 12]
SNR_DB_RANGE  = np.arange(0, 31, 2)          # 0–30 dB, step 2 dB
AWGN_TRIALS   = 5                              # repeated BER trials per SNR point
RAYLEIGH_TRIALS = 5

# Demodulated column suffix per scheme
DEMOD_SUFFIX = {
    "AM":  "_am_demodulated",
    "FM":  "_fm_demodulated",
    "ASK": "_ask_recovered",
    "FSK": "_fsk_recovered",
    "PSK": "_psk_recovered",
}


# ─────────────────────────────────────────────
#  HELPER: find demodulated signal columns
# ─────────────────────────────────────────────
def get_demod_cols(df: pd.DataFrame, scheme: str) -> list[str]:
    suffix = DEMOD_SUFFIX[scheme]
    return [c for c in df.columns if c.endswith(suffix)]


# ─────────────────────────────────────────────
#  QUANTIZATION & PCM
# ─────────────────────────────────────────────
def quantize(signal: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """
    Mid-tread uniform quantizer.
    Returns (quantized_signal_normalised, SQNR_dB).
    Input signal assumed in [0, 1].
    """
    levels = 2 ** bits
    step   = 1.0 / levels
    q = np.clip(np.floor(signal / step) / levels, 0.0, 1.0 - step)
    noise  = signal - q
    signal_power = np.mean(signal ** 2)
    noise_power  = np.mean(noise  ** 2)
    if noise_power < 1e-15:
        sqnr_db = 120.0   # essentially perfect
    else:
        sqnr_db = 10.0 * np.log10(signal_power / noise_power)
    return q, sqnr_db


def signal_to_pcm_bits(quantized_signal: np.ndarray, bits: int) -> np.ndarray:
    """Convert normalised quantised signal to a flat bitstream (MSB first)."""
    levels = 2 ** bits
    codes  = np.clip(np.round(quantized_signal * (levels - 1)).astype(int), 0, levels - 1)
    rows   = np.unpackbits(
        codes.astype(np.uint16).view(np.uint8).reshape(-1, 2)[:, ::-1],
        axis=1
    )[:, 16 - bits:]       # keep only the `bits` LSBs
    return rows.flatten()


# ─────────────────────────────────────────────
#  LINE CODING
# ─────────────────────────────────────────────
def nrz_encode(bits: np.ndarray) -> np.ndarray:
    """NRZ-L: 0 → -1, 1 → +1."""
    return 2 * bits.astype(float) - 1.0


def manchester_encode(bits: np.ndarray) -> np.ndarray:
    """IEEE 802.3 Manchester: each bit becomes two half-period chips.
       0 → [+1, -1],  1 → [-1, +1]."""
    chips = np.empty(len(bits) * 2)
    chips[0::2] = np.where(bits == 0,  1.0, -1.0)
    chips[1::2] = np.where(bits == 0, -1.0,  1.0)
    return chips


def nrz_decode(symbols: np.ndarray) -> np.ndarray:
    return (symbols >= 0).astype(int)


def manchester_decode(chips: np.ndarray) -> np.ndarray:
    """Majority vote on first half-chip: +1 → 0, -1 → 1."""
    n = len(chips) // 2
    first = chips[:2 * n:2]
    return (first < 0).astype(int)


# ─────────────────────────────────────────────
#  CHANNEL MODELS
# ─────────────────────────────────────────────
def awgn_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    snr_lin = 10 ** (snr_db / 10.0)
    p_signal = np.mean(signal ** 2)
    sigma    = np.sqrt(p_signal / (2.0 * snr_lin))
    return signal + sigma * np.random.randn(*signal.shape)


def rayleigh_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Flat Rayleigh fading + AWGN."""
    snr_lin  = 10 ** (snr_db / 10.0)
    p_signal = np.mean(signal ** 2)
    sigma_n  = np.sqrt(p_signal / (2.0 * snr_lin))
    # Rayleigh fading coefficients (envelope = |h|, h ~ CN(0,1))
    h = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) / np.sqrt(2)
    faded = np.abs(h) * signal
    return faded + sigma_n * np.random.randn(*signal.shape)


def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 0.5
    return float(np.sum(tx_bits[:n] != rx_bits[:n])) / n


# ─────────────────────────────────────────────
#  INTEGRITY CHECKS
# ─────────────────────────────────────────────
def add_even_parity(bits: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Append one even-parity bit after every `block_size` data bits."""
    n_blocks = len(bits) // block_size
    blocks   = bits[:n_blocks * block_size].reshape(n_blocks, block_size)
    parity   = (np.sum(blocks, axis=1) % 2).reshape(-1, 1)
    return np.hstack([blocks, parity]).flatten()


def verify_parity(bits: np.ndarray, block_size: int = 8) -> dict:
    frame_size = block_size + 1
    n_frames   = len(bits) // frame_size
    frames     = bits[:n_frames * frame_size].reshape(n_frames, frame_size)
    data_part  = frames[:, :block_size]
    parity_bit = frames[:, block_size]
    computed   = np.sum(data_part, axis=1) % 2
    errors     = int(np.sum(computed != parity_bit))
    return {"total_frames": n_frames, "parity_errors": errors, "parity_error_rate": errors / max(n_frames, 1)}


def fletcher16_checksum(bits: np.ndarray) -> int:
    """Fletcher-16 checksum over 8-bit words formed from the bitstream."""
    n_bytes = len(bits) // 8
    if n_bytes == 0:
        return 0
    data   = np.packbits(bits[:n_bytes * 8])
    sum1, sum2 = 0, 0
    for byte in data:
        sum1 = (sum1 + int(byte)) % 255
        sum2 = (sum2 + sum1) % 255
    return (sum2 << 8) | sum1


def verify_checksum(tx_bits: np.ndarray, rx_bits: np.ndarray) -> dict:
    tx_cs = fletcher16_checksum(tx_bits)
    rx_cs = fletcher16_checksum(rx_bits)
    return {"tx_checksum": tx_cs, "rx_checksum": rx_cs, "match": tx_cs == rx_cs}


# ─────────────────────────────────────────────
#  BER SIMULATION FOR ONE SIGNAL VECTOR
# ─────────────────────────────────────────────
def simulate_ber_curve(bits: np.ndarray, coding: str, channel: str,
                        snr_range: np.ndarray, trials: int) -> list[float]:
    """Return mean BER at each SNR point."""
    encode_fn = nrz_encode if coding == "NRZ" else manchester_encode
    decode_fn = nrz_decode if coding == "NRZ" else manchester_decode
    ber_curve = []
    for snr_db in snr_range:
        trial_bers = []
        for _ in range(trials):
            encoded = encode_fn(bits)
            if channel == "AWGN":
                noisy = awgn_channel(encoded, snr_db)
            else:
                noisy = rayleigh_channel(encoded, snr_db)
            decoded = decode_fn(noisy)
            trial_bers.append(compute_ber(bits, decoded))
        ber_curve.append(float(np.mean(trial_bers)))
    return ber_curve


# ─────────────────────────────────────────────
#  MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────
def process_all(data_dir: Path, out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging setup ──────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("digital_telemetry")

    results       = {}          # top-level dict → JSON
    monitor_lines = []          # lines → monitoring log

    # ── Iterate stations ───────────────────────────────────────────
    for station in STATIONS:
        log.info(f"Processing station: {station}")
        results[station] = {}

        station_monitor_rows = []   # collect rows across all schemes for this station

        for scheme in SCHEMES:
            filtered_path  = data_dir / f"{station}_filtered_{scheme}_demod.csv"
            features_path  = data_dir / f"{station}_features_{scheme}_demod.csv"

            if not filtered_path.exists():
                log.warning(f"  Missing: {filtered_path.name}")
                continue

            df_filt = pd.read_csv(filtered_path, parse_dates=["timestamp"])
            df_feat = pd.read_csv(features_path) if features_path.exists() else None

            demod_cols = get_demod_cols(df_filt, scheme)
            if not demod_cols:
                log.warning(f"  No demod cols for {station}/{scheme}")
                continue

            log.info(f"  Scheme={scheme}  demod_cols={len(demod_cols)}")

            scheme_result = {}

            # ── 1. Quantize & PCM at 8, 10, 12 bits ─────────────
            sqnr_table = {}           # bit_depth → {col → sqnr_db}
            best_bits_per_col = {}    # col → best bit depth

            for col in demod_cols:
                signal = df_filt[col].fillna(0).values.astype(float)
                # Normalise to [0,1]
                sig_min, sig_max = signal.min(), signal.max()
                rng = sig_max - sig_min
                norm_signal = (signal - sig_min) / rng if rng > 1e-10 else np.zeros_like(signal)

                col_sqnrs = {}
                for bits in BIT_DEPTHS:
                    _, sqnr = quantize(norm_signal, bits)
                    col_sqnrs[bits] = round(sqnr, 3)
                sqnr_table[col] = col_sqnrs
                best_bits_per_col[col] = max(col_sqnrs, key=col_sqnrs.get)

            scheme_result["sqnr_by_col_and_bits"] = sqnr_table
            scheme_result["best_bit_depth_per_col"] = best_bits_per_col

            # ── 2. BER simulation (use representative column) ─────
            #    Use the first demod column; report per-station summary
            rep_col  = demod_cols[0]
            rep_bits_depth = best_bits_per_col[rep_col]
            signal   = df_filt[rep_col].fillna(0).values.astype(float)
            sig_min, sig_max = signal.min(), signal.max()
            rng = sig_max - sig_min
            norm_signal = (signal - sig_min) / rng if rng > 1e-10 else np.zeros_like(signal)
            q_sig, _ = quantize(norm_signal, rep_bits_depth)
            pcm_bits = signal_to_pcm_bits(q_sig, rep_bits_depth)

            ber_data = {}
            for coding in ["NRZ", "Manchester"]:
                ber_data[coding] = {}
                for channel in ["AWGN", "Rayleigh"]:
                    trials = AWGN_TRIALS if channel == "AWGN" else RAYLEIGH_TRIALS
                    curve  = simulate_ber_curve(pcm_bits, coding, channel,
                                                SNR_DB_RANGE, trials)
                    ber_data[coding][channel] = {
                        "snr_db":    SNR_DB_RANGE.tolist(),
                        "ber_curve": curve,
                        "ber_at_10db": curve[SNR_DB_RANGE.tolist().index(10)],
                    }

            scheme_result["ber"] = ber_data

            # ── 3. Line coding recommendation ─────────────────────
            nrz_ber10  = ber_data["NRZ"]["AWGN"]["ber_at_10db"]
            man_ber10  = ber_data["Manchester"]["AWGN"]["ber_at_10db"]
            if man_ber10 <= nrz_ber10:
                recommended = "Manchester"
                reason = (
                    "Manchester achieved equal or lower BER at 10 dB SNR (AWGN). "
                    "Its self-clocking property and guaranteed transitions also improve "
                    "synchronisation for the downstream monitoring pipeline."
                )
            else:
                recommended = "NRZ"
                reason = (
                    "NRZ achieved lower BER at 10 dB SNR (AWGN) for this station/scheme "
                    "and requires half the bandwidth of Manchester encoding, which is "
                    "advantageous given the low-rate telemetry context."
                )
            scheme_result["line_coding_recommendation"] = {
                "recommended": recommended,
                "reason": reason,
                "nrz_ber_at_10db":  round(nrz_ber10, 6),
                "man_ber_at_10db":  round(man_ber10, 6),
            }

            # ── 4. Integrity checks ────────────────────────────────
            # Use NRZ+AWGN at 10 dB as the simulated received bitstream
            nrz_enc = nrz_encode(pcm_bits)
            rx_noisy = awgn_channel(nrz_enc, 10.0)
            rx_bits  = nrz_decode(rx_noisy)

            parity_result   = verify_parity(rx_bits)
            checksum_result = verify_checksum(pcm_bits, rx_bits)

            scheme_result["integrity"] = {
                "parity":    parity_result,
                "checksum":  checksum_result,
                "ber_summary": {
                    "total_bits":   int(len(pcm_bits)),
                    "bit_errors":   int(np.sum(pcm_bits != rx_bits[:len(pcm_bits)])),
                    "raw_ber":      round(compute_ber(pcm_bits, rx_bits), 6),
                }
            }

            results[station][scheme] = scheme_result

            # ── 5. Collect monitoring rows ──────────────────────────
            if df_feat is not None:
                for _, row in df_feat.iterrows():
                    station_monitor_rows.append({
                        "station":   station,
                        "scheme":    scheme,
                        "timestamp": str(row.get("segment_start", "")),
                        **{k: v for k, v in row.items() if k not in ["segment_start","segment_end"]},
                    })
            else:
                # Fall back to filtered data rows
                for _, row in df_filt.iterrows():
                    station_monitor_rows.append({
                        "station":  station,
                        "scheme":   scheme,
                        "timestamp": str(row["timestamp"]),
                        **{k: v for k, v in row.items() if k != "timestamp"},
                    })

        # ── Sort monitoring rows chronologically ───────────────────
        station_monitor_rows.sort(key=lambda x: x.get("timestamp",""))
        monitor_lines.extend(station_monitor_rows)

    # ─────────────────────────────────────────────────────────────
    #  OUTPUT 1 – MONITORING STREAM LOG
    # ─────────────────────────────────────────────────────────────
    log_path = out_dir / "monitoring_stream.log"
    with open(log_path, "w") as f:
        f.write("# TELE 523 Group 1 – Monitoring Lead Stream Log\n")
        f.write("# Format: JSON-lines.  Each line is one observation.\n")
        f.write("# Fields: timestamp, station, scheme, + all feature/signal columns\n")
        f.write("# REPLAY_INTERVAL_HINT: 2.0s (real-time) | 0.1s (fast-replay)\n")
        f.write("# Use --fast flag in your Streamlit app to switch between modes.\n")
        f.write("#\n")
        written = 0
        for row in monitor_lines:
            f.write(json.dumps(row, default=str) + "\n")
            written += 1
    log.info(f"Monitoring log written: {log_path}  ({written} lines)")

    # ─────────────────────────────────────────────────────────────
    #  OUTPUT 2 – DIGITAL TELEMETRY JSON
    # ─────────────────────────────────────────────────────────────
    json_path = out_dir / "digital_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "project": "TELE 523 – Group 1 – Industrial Machine Condition Monitoring",
                    "author_role": "Digital Telemetry Lead",
                    "bit_depths": BIT_DEPTHS,
                    "snr_range_db": [int(SNR_DB_RANGE[0]), int(SNR_DB_RANGE[-1])],
                    "channel_models": ["AWGN", "Rayleigh"],
                    "line_coding_schemes": ["NRZ", "Manchester"],
                    "modulation_schemes": SCHEMES,
                    "stations": STATIONS,
                    "integrity_checks": ["even_parity", "fletcher16_checksum", "ber"],
                },
                "results": results,
            },
            f,
            indent=2,
            default=str
        )
    log.info(f"Digital telemetry JSON written: {json_path}")

    # ─────────────────────────────────────────────────────────────
    #  OUTPUT 3 – DIAGNOSTIC PLOTS
    # ─────────────────────────────────────────────────────────────
    _plot_ber_curves(results, plots_dir, log)
    _plot_sqnr_bars(results, plots_dir, log)
    _plot_eye_diagrams(data_dir, plots_dir, log)
    _plot_ber_scheme_comparison(results, plots_dir, log)

    log.info("All outputs complete.")
    return log_path, json_path


# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────
SCHEME_COLORS = {"AM": "#1f77b4", "FM": "#ff7f0e", "ASK": "#2ca02c",
                 "FSK": "#d62728", "PSK": "#9467bd"}
CODING_LS     = {"NRZ": "-", "Manchester": "--"}
CHANNEL_MARK  = {"AWGN": "o", "Rayleigh": "s"}


def _plot_ber_curves(results: dict, plots_dir: Path, log):
    """One figure per station: BER vs SNR for all 5 schemes, NRZ, AWGN."""
    for station in STATIONS:
        if station not in results:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for scheme in SCHEMES:
            if scheme not in results[station]:
                continue
            ber_data = results[station][scheme].get("ber", {})
            if "NRZ" not in ber_data or "AWGN" not in ber_data["NRZ"]:
                continue
            entry = ber_data["NRZ"]["AWGN"]
            ax.semilogy(entry["snr_db"], np.clip(entry["ber_curve"], 1e-6, 1),
                        color=SCHEME_COLORS[scheme], label=scheme, linewidth=2)
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title(f"{station} – BER vs SNR  (NRZ / AWGN)", fontsize=13)
        ax.legend(title="Modulation", fontsize=9)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.set_xlim(0, 30)
        ax.set_ylim(1e-6, 1)
        fig.tight_layout()
        path = plots_dir / f"{station}_BER_curves_NRZ_AWGN.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"  Plot: {path.name}")


def _plot_ber_scheme_comparison(results: dict, plots_dir: Path, log):
    """
    One figure per station: 4-panel grid (NRZ/Manchester × AWGN/Rayleigh),
    all 5 modulation schemes overlaid, fixed SNR=10 dB bar comparison inset.
    """
    for station in STATIONS:
        if station not in results:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        panels = [("NRZ","AWGN"), ("NRZ","Rayleigh"),
                  ("Manchester","AWGN"), ("Manchester","Rayleigh")]
        for idx, (coding, channel) in enumerate(panels):
            ax = axes[idx // 2][idx % 2]
            for scheme in SCHEMES:
                if scheme not in results[station]:
                    continue
                try:
                    entry = results[station][scheme]["ber"][coding][channel]
                    ax.semilogy(entry["snr_db"],
                                np.clip(entry["ber_curve"], 1e-6, 1),
                                color=SCHEME_COLORS[scheme],
                                label=scheme, linewidth=1.8)
                except KeyError:
                    pass
            ax.set_title(f"{coding} / {channel}", fontsize=11)
            ax.set_ylabel("BER")
            ax.set_xlabel("SNR (dB)")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8, title="Scheme")
            ax.set_xlim(0, 30)
            ax.set_ylim(1e-6, 1)
        fig.suptitle(f"{station} – BER Comparison Across Modulation Schemes", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = plots_dir / f"{station}_BER_all_panels.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"  Plot: {path.name}")

    # ── Cross-station BER bar chart at SNR=10 dB (NRZ, AWGN) ──
    snr10_idx = SNR_DB_RANGE.tolist().index(10)
    data_bar  = {scheme: [] for scheme in SCHEMES}
    valid_stations = []
    for station in STATIONS:
        if station not in results:
            continue
        valid_stations.append(station)
        for scheme in SCHEMES:
            try:
                val = results[station][scheme]["ber"]["NRZ"]["AWGN"]["ber_curve"][snr10_idx]
            except (KeyError, IndexError):
                val = np.nan
            data_bar[scheme].append(val)

    n_stations = len(valid_stations)
    x = np.arange(n_stations)
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, scheme in enumerate(SCHEMES):
        vals = [v if not np.isnan(v) else 0 for v in data_bar[scheme]]
        ax.bar(x + i * width, vals, width, label=scheme, color=SCHEME_COLORS[scheme])
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(valid_stations, rotation=20, ha="right")
    ax.set_ylabel("BER @ 10 dB SNR")
    ax.set_title("Cross-Station BER Comparison at SNR=10 dB (NRZ / AWGN)", fontsize=13)
    ax.legend(title="Modulation", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    path = plots_dir / "CROSS_STATION_BER_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Plot: {path.name}")


def _plot_sqnr_bars(results: dict, plots_dir: Path, log):
    """SQNR vs bit depth per station (averaged across all signal columns & schemes)."""
    for station in STATIONS:
        if station not in results:
            continue
        sqnr_by_bits: dict[int, list] = {b: [] for b in BIT_DEPTHS}
        for scheme in SCHEMES:
            if scheme not in results[station]:
                continue
            for col, col_sq in results[station][scheme].get("sqnr_by_col_and_bits", {}).items():
                for b in BIT_DEPTHS:
                    if b in col_sq:
                        sqnr_by_bits[b].append(col_sq[b])

        means = [np.mean(sqnr_by_bits[b]) if sqnr_by_bits[b] else 0 for b in BIT_DEPTHS]
        stds  = [np.std(sqnr_by_bits[b])  if sqnr_by_bits[b] else 0 for b in BIT_DEPTHS]

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#4e79a7", "#f28e2b", "#59a14f"]
        bars = ax.bar([str(b) for b in BIT_DEPTHS], means, yerr=stds,
                      color=colors, capsize=5, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f} dB", ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("Quantisation Bit Depth", fontsize=12)
        ax.set_ylabel("Mean SQNR (dB)", fontsize=12)
        ax.set_title(f"{station} – SQNR vs Bit Depth\n(mean ± σ over all signal cols & schemes)", fontsize=11)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()
        path = plots_dir / f"{station}_SQNR_vs_bitdepth.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"  Plot: {path.name}")

    # ── Theory overlay: SQNR ≈ 6.02n + 1.76 dB ──
    fig, ax = plt.subplots(figsize=(8, 4))
    n_bits   = np.array(BIT_DEPTHS)
    theory   = 6.02 * n_bits + 1.76
    ax.plot(n_bits, theory, "k--", linewidth=2, label="Theory: 6.02n+1.76 dB")
    for station in STATIONS:
        if station not in results:
            continue
        means_s = []
        for b in BIT_DEPTHS:
            vals = []
            for scheme in SCHEMES:
                for col, col_sq in results.get(station, {}).get(scheme, {}).get("sqnr_by_col_and_bits", {}).items():
                    if b in col_sq:
                        vals.append(col_sq[b])
            means_s.append(np.mean(vals) if vals else np.nan)
        ax.plot(n_bits, means_s, "o-", label=station, linewidth=1.5, markersize=5)
    ax.set_xlabel("Bit Depth (n)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("SQNR vs Bit Depth – All Stations + Theory")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    path = plots_dir / "ALL_STATIONS_SQNR_theory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Plot: {path.name}")


def _plot_eye_diagrams(data_dir: Path, plots_dir: Path, log):
    """NRZ vs Manchester eye diagram for the first available signal of each station."""
    for station in STATIONS:
        path_am = data_dir / f"{station}_filtered_AM_demod.csv"
        if not path_am.exists():
            continue
        df = pd.read_csv(path_am)
        demod_cols = [c for c in df.columns if c.endswith("_am_demodulated")]
        if not demod_cols:
            continue
        col     = demod_cols[0]
        signal  = df[col].fillna(0).values.astype(float)
        rng = signal.max() - signal.min()
        norm    = (signal - signal.min()) / rng if rng > 1e-10 else np.zeros_like(signal)
        q, _    = quantize(norm, 8)
        pcm     = signal_to_pcm_bits(q, 8)

        nrz_sym = nrz_encode(pcm)
        man_sym = manchester_encode(pcm)

        # Add AWGN at 10 dB
        nrz_noisy = awgn_channel(nrz_sym, 10.0)
        man_noisy = awgn_channel(man_sym, 10.0)

        # Build eye diagram: fold waveform every T_eye samples
        def eye_diagram(waveform, sps=2, n_traces=200):
            """Return (x_grid, y_traces) for an eye plot."""
            period  = sps
            n_eyes  = min(n_traces, len(waveform) // period)
            traces  = [waveform[i * period:(i + 1) * period] for i in range(n_eyes)]
            valid   = [t for t in traces if len(t) == period]
            x       = np.linspace(0, 1, period)
            return x, valid

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, label, wfm in zip(axes, ["NRZ", "Manchester"],
                                         [nrz_noisy, man_noisy]):
            sps   = 1 if label == "NRZ" else 2
            x, traces = eye_diagram(wfm, sps=sps, n_traces=300)
            for t in traces:
                ax.plot(x, t, color="#1f77b4", alpha=0.05, linewidth=0.8)
            ax.set_title(f"{label} Eye Diagram\n{station} / AM / 8-bit PCM / AWGN 10 dB", fontsize=10)
            ax.set_xlabel("Time (normalised)", fontsize=10)
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle=":", alpha=0.5)
        fig.suptitle(f"{station} – NRZ vs Manchester Eye Diagrams", fontsize=12, fontweight="bold")
        fig.tight_layout()
        path = plots_dir / f"{station}_eye_diagram.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"  Plot: {path.name}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TELE 523 Digital Telemetry Lead Script")
    parser.add_argument("--data-dir", default="./modulation_output/raw",
                        help="Path to folder containing modulation lead output CSV files")
    parser.add_argument("--out-dir",  default="./results/digital_telemetry",
                        help="Path to folder where outputs will be saved")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path, json_path = process_all(data_dir, out_dir)

    print("\n" + "="*60)
    print("OUTPUTS")
    print("="*60)
    print(f"  Monitoring log :  {log_path}")
    print(f"  Telemetry JSON :  {json_path}")
    print(f"  Plots folder   :  {out_dir / 'plots'}")
    print("="*60)