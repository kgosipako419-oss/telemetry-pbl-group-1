"""
generate_demodulation_outputs.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Produces the Student 4 handoff files — 35 CSVs (5 schemes × 7 stations).

Imports all pipeline functions from demodulation.py.
No modulation or demodulation math lives here — this script only handles
loading inputs, calling pipelines, and saving outputs.

Pipeline for each file:
  baseband signal → modulate → AWGN channel → DEMODULATE → save CSV

Output location: results/demodulation/output/
Output files (35 total):
  {STATION}_AM_demod.csv   — AM  recovered analog signal [0,1]
  {STATION}_FM_demod.csv   — FM  recovered analog signal [0,1]
  {STATION}_ASK_demod.csv  — ASK recovered bits {0,1} + original bits
  {STATION}_FSK_demod.csv  — FSK recovered bits {0,1} + original bits
  {STATION}_PSK_demod.csv  — PSK recovered bits {0,1} + original bits

Column layout per file type:
  Analog (AM, FM):
    timestamp | {col}_baseband | {col}_{scheme}_demodulated | ... | current_state_binary

  Digital (ASK, FSK, PSK):
    timestamp | {col}_baseband | {col}_original_bits | {col}_{scheme}_recovered | ... | current_state_binary

What Student 4 does with these files:
  AM / FM  → uniform quantization (e.g. 8-bit PCM) on *_demodulated columns
  ASK / FSK / PSK → line coding (NRZ-L, Manchester) on *_recovered columns,
                    then PCM bitstream integrity checks
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Import pipeline functions from demodulation.py ────────────────────────────
try:
    from demodulation import (
        SIGNAL_COLS, PROCESSED_PATH,
        normalise, _expand_bits,
        pipeline_am, pipeline_fm,
        pipeline_ask, pipeline_fsk, pipeline_psk,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from demodulation import (
        SIGNAL_COLS, PROCESSED_PATH,
        normalise, _expand_bits,
        pipeline_am, pipeline_fm,
        pipeline_ask, pipeline_fsk, pipeline_psk,
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "results", "demodulation", "output")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# OUTPUT BUILDERS — one function per scheme
# Each builds a dict of columns for that scheme's output CSV.
# =============================================================================

def _build_am(df, sig_cols, ts, labels):
    """Build column dict for {STATION}_AM_demod.csv."""
    out = {"timestamp": ts, "current_state_binary": labels}
    metrics = []
    for col in sig_cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 30:
            continue
        _, _, dem, snr = pipeline_am(x)
        out[f"{col}_baseband"]        = np.round(normalise(x), 6)
        out[f"{col}_am_demodulated"]  = np.round(dem, 6)
        metrics.append((col, "AM", snr, None))
        print(f"    AM  · {col:30s}  SNR = {snr:6.2f} dB")
    return out, metrics


def _build_fm(df, sig_cols, ts, labels):
    """Build column dict for {STATION}_FM_demod.csv."""
    out = {"timestamp": ts, "current_state_binary": labels}
    metrics = []
    for col in sig_cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 30:
            continue
        _, _, dem, snr = pipeline_fm(x)
        out[f"{col}_baseband"]        = np.round(normalise(x), 6)
        out[f"{col}_fm_demodulated"]  = np.round(dem, 6)
        metrics.append((col, "FM", snr, None))
        print(f"    FM  · {col:30s}  SNR = {snr:6.2f} dB")
    return out, metrics


def _build_ask(df, sig_cols, ts, labels):
    """Build column dict for {STATION}_ASK_demod.csv."""
    out = {"timestamp": ts, "current_state_binary": labels}
    metrics = []
    for col in sig_cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 30:
            continue
        n = len(x)
        _, _, rec, env, snr, ber_val, orig = pipeline_ask(x)
        out[f"{col}_baseband"]        = np.round(normalise(x), 6)
        out[f"{col}_original_bits"]   = _expand_bits(orig, n).astype(int)
        out[f"{col}_ask_recovered"]   = _expand_bits(rec,  n).astype(int)
        metrics.append((col, "ASK", snr, ber_val))
        print(f"    ASK · {col:30s}  SNR = {snr:6.2f} dB  BER = {ber_val:.5f}")
    return out, metrics


def _build_fsk(df, sig_cols, ts, labels):
    """Build column dict for {STATION}_FSK_demod.csv."""
    out = {"timestamp": ts, "current_state_binary": labels}
    metrics = []
    for col in sig_cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 30:
            continue
        n = len(x)
        _, _, rec, snr, ber_val, orig = pipeline_fsk(x)
        out[f"{col}_baseband"]        = np.round(normalise(x), 6)
        out[f"{col}_original_bits"]   = _expand_bits(orig, n).astype(int)
        out[f"{col}_fsk_recovered"]   = _expand_bits(rec,  n).astype(int)
        metrics.append((col, "FSK", snr, ber_val))
        print(f"    FSK · {col:30s}  SNR = {snr:6.2f} dB  BER = {ber_val:.5f}")
    return out, metrics


def _build_psk(df, sig_cols, ts, labels):
    """Build column dict for {STATION}_PSK_demod.csv."""
    out = {"timestamp": ts, "current_state_binary": labels}
    metrics = []
    for col in sig_cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 30:
            continue
        n = len(x)
        _, _, rec, snr, ber_val, orig = pipeline_psk(x)
        out[f"{col}_baseband"]        = np.round(normalise(x), 6)
        out[f"{col}_original_bits"]   = _expand_bits(orig, n).astype(int)
        out[f"{col}_psk_recovered"]   = _expand_bits(rec,  n).astype(int)
        metrics.append((col, "PSK", snr, ber_val))
        print(f"    PSK · {col:30s}  SNR = {snr:6.2f} dB  BER = {ber_val:.5f}")
    return out, metrics


def _save(data_dict, path):
    """Order columns (timestamp first, label last) and save CSV."""
    df_out   = pd.DataFrame(data_dict)
    mid_cols = [c for c in df_out.columns
                if c not in ("timestamp", "current_state_binary")]
    df_out   = df_out[["timestamp"] + mid_cols + ["current_state_binary"]]
    df_out.to_csv(path, index=False)
    return df_out.shape


# =============================================================================
# MAIN
# =============================================================================

def generate_demodulation_outputs():
    print("=" * 70)
    print("TELE 523 · Student 3 — Demodulation Output Generator")
    print("Producing 35 handoff CSV files for Student 4")
    print("Each station × each scheme × each signal column — independently")
    print("=" * 70)

    SCHEMES = [
        ("AM",  _build_am,  "AM_demod"),
        ("FM",  _build_fm,  "FM_demod"),
        ("ASK", _build_ask, "ASK_demod"),
        ("FSK", _build_fsk, "FSK_demod"),
        ("PSK", _build_psk, "PSK_demod"),
    ]

    all_metrics = []
    total_files = 0

    for station in sorted(SIGNAL_COLS.keys()):
        fpath = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
        if not os.path.exists(fpath):
            print(f"\n[SKIP] {station} — {station}_filtered.csv not found")
            continue

        df       = pd.read_csv(fpath, parse_dates=["timestamp"])
        sig_cols = [c for c in SIGNAL_COLS[station] if c in df.columns]
        labels   = (df["current_state_binary"].fillna(0).astype(int)
                    if "current_state_binary" in df.columns
                    else pd.Series(np.zeros(len(df), dtype=int)))
        ts = df["timestamp"]

        print(f"\n-- {station} " + "-" * 50)

        for scheme_name, builder_fn, suffix in SCHEMES:
            print(f"  [{scheme_name}]")
            data_dict, metrics = builder_fn(df, sig_cols, ts, labels)

            out_path    = os.path.join(OUT_DIR, f"{station}_{suffix}.csv")
            rows, cols  = _save(data_dict, out_path)
            total_files += 1

            print(f"  --> Saved {station}_{suffix}.csv  ({rows} rows × {cols} cols)")

            for col, scheme, snr, ber_val in metrics:
                all_metrics.append({
                    "station"    : station,
                    "signal_col" : col,
                    "scheme"     : scheme,
                    "SNR_dB"     : round(snr, 3),
                    "BER"        : round(ber_val, 5) if ber_val is not None else None,
                })

    # ── Save metrics summary ───────────────────────────────────────────────────
    metrics_df  = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(
        os.path.dirname(OUT_DIR), "demodulation_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DONE — {total_files} demodulation output files written to:")
    print(f"  {OUT_DIR}")
    print(f"\nMetrics CSV saved → {metrics_path}")
    print("\nSNR summary (mean per station × scheme):")
    pivot_snr = metrics_df.pivot_table(
        values="SNR_dB", index="station", columns="scheme", aggfunc="mean"
    ).round(2)
    print(pivot_snr.to_string())

    print("\nBER summary (digital schemes, mean per station):")
    pivot_ber = metrics_df[metrics_df["BER"].notna()].pivot_table(
        values="BER", index="station", columns="scheme", aggfunc="mean"
    ).round(5)
    print(pivot_ber.to_string())

    print("\n✔  Student 4 handoff complete.")
    print("   AM/FM files  → apply uniform quantization (PCM)")
    print("   ASK/FSK/PSK  → apply NRZ / Manchester line coding")


if __name__ == "__main__":
    generate_demodulation_outputs()