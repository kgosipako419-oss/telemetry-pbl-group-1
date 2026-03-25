"""
modulation.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Implements AM, FM, ASK, FSK, and PSK modulation and demodulation on the
FIR-filtered signals and feature vectors produced by Student 2 (Signal Processing Lead).

Inputs (from data/processed/):
  *_filtered.csv  — FIR-filtered time-series (baseband signal)
  *_features.csv  — segment-level feature vectors (RMS, PSD, FFT, etc.)

Outputs (to results/modulation/):
  *_modulation_results.csv  — SNR and BER per scheme per station
  *_modulation_plots.png    — waveform comparison plots per station
  modulation_summary.png    — cross-station BER comparison table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR        = os.path.join(BASE_DIR, "results", "modulation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global carrier / channel parameters ───────────────────────────────────────
FS        = 0.5          # Hz — inherited from signal processing lead (sampling rate)
FC        = 0.05         # Hz — carrier frequency (must be < Nyquist = 0.25 Hz)
NOISE_STD = 0.05         # AWGN standard deviation for channel simulation
BIT_RATE  = 0.05         # bits per sample (for digital schemes)

# Stations and their primary motor/speed signal columns (baseband source)
SIGNAL_COLS = {
    "EC_1"  : ["i3_photoresistor", "current_task_duration"],
    "HBW_1" : ["m1_speed", "m2_speed", "m3_speed", "m4_speed", "current_task_duration"],
    "MM_1"  : ["m1_speed", "m2_speed", "m3_speed", "current_task_duration"],
    "OV_1"  : ["m1_speed", "current_task_duration"],
    "SM_1"  : ["m1_speed", "current_task_duration"],
    "VGR_1" : ["m1_speed", "m2_speed", "m3_speed", "current_task_duration"],
    "WT_1"  : ["m2_speed", "current_task_duration"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def time_axis(n, fs=FS):
    """Return a time array of n samples at rate fs (Hz)."""
    return np.arange(n) / fs


def carrier(n, fc=FC, fs=FS):
    """Return a cosine carrier of length n."""
    t = time_axis(n, fs)
    return np.cos(2 * np.pi * fc * t)


def awgn(signal, noise_std=NOISE_STD, seed=42):
    """Add Additive White Gaussian Noise to a signal."""
    rng = np.random.default_rng(seed)
    return signal + rng.normal(0, noise_std, size=len(signal))


def snr_db(clean, noisy):
    """Compute Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean((clean - noisy) ** 2)
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return 10 * np.log10(signal_power / noise_power)


def ber(original_bits, recovered_bits):
    """
    Bit Error Rate — fraction of bits that differ.
    Both inputs must be integer arrays of the same length.
    """
    n = min(len(original_bits), len(recovered_bits))
    return np.sum(original_bits[:n] != recovered_bits[:n]) / n


def signal_to_bits(signal):
    """
    Convert a normalised [0,1] signal to a binary sequence by thresholding at 0.5.
    Used as the 'digital message' for ASK / FSK / PSK.
    """
    clipped = np.clip(signal, 0, 1)
    return (clipped > 0.5).astype(int)


def normalise(signal):
    """Min-max normalise a signal to [0, 1]. Safe against constant signals."""
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-12:
        return np.zeros_like(signal, dtype=float)
    return (signal - lo) / (hi - lo)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALOG MODULATION: AM and FM
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2.1 Amplitude Modulation (AM) ────────────────────────────────────────────

def am_modulate(baseband, fc=FC, fs=FS, mod_index=0.8):
    """
    AM modulation:  s(t) = [1 + m·x(t)] · cos(2π·fc·t)
    where x(t) is the normalised baseband signal and m is the modulation index.
    """
    x = normalise(baseband)
    t = time_axis(len(x), fs)
    c = np.cos(2 * np.pi * fc * t)
    return (1 + mod_index * x) * c


def am_demodulate(modulated, fc=FC, fs=FS):
    """
    AM demodulation via envelope detection using the Hilbert transform.
    Recovers the envelope |analytic_signal(s(t))| then removes the DC offset.
    """
    analytic = hilbert(modulated)
    envelope = np.abs(analytic)
    # Remove DC offset introduced by the carrier
    envelope -= np.mean(envelope)
    return envelope


# ── 2.2 Frequency Modulation (FM) ────────────────────────────────────────────

def fm_modulate(baseband, fc=FC, fs=FS, kf=0.1):
    """
    FM modulation:  s(t) = cos(2π·fc·t + 2π·kf·∫x(τ)dτ)
    kf is the frequency deviation constant (Hz per unit amplitude).
    Carson's rule bandwidth ≈ 2(kf·A + W) where A=peak amplitude, W=message BW.
    """
    x = normalise(baseband)
    t = time_axis(len(x), fs)
    # Cumulative integral via cumulative sum (trapezoidal approximation at 1/fs)
    integral = np.cumsum(x) / fs
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral)


def fm_demodulate(modulated, fc=FC, fs=FS):
    """
    FM demodulation via instantaneous frequency estimation.
    Differentiates the phase of the analytic signal.
    """
    analytic  = hilbert(modulated)
    phase     = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi / fs)
    # Pad to original length
    inst_freq = np.append(inst_freq, inst_freq[-1])
    return inst_freq - fc   # remove carrier frequency offset


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DIGITAL MODULATION: ASK, FSK, PSK
# ═══════════════════════════════════════════════════════════════════════════════

def _samples_per_bit(n_samples, n_bits):
    """Return integer samples-per-bit given total samples and bit count."""
    return max(1, n_samples // max(1, n_bits))


# ── 3.1 Amplitude Shift Keying (ASK / OOK) ───────────────────────────────────

def ask_modulate(bits, n_samples, fc=FC, fs=FS):
    """
    Binary ASK (On-Off Keying):
      bit=1 → carrier at full amplitude
      bit=0 → carrier at zero amplitude
    """
    spb = _samples_per_bit(n_samples, len(bits))
    t   = time_axis(n_samples, fs)
    c   = np.cos(2 * np.pi * fc * t)
    env = np.repeat(bits.astype(float), spb)[:n_samples]
    # Pad if needed
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)), constant_values=env[-1])
    return env * c


def ask_demodulate(modulated, bits_orig, fc=FC, fs=FS):
    """
    ASK demodulation via envelope detection + threshold at 0.5.
    Returns recovered bit array.
    """
    n   = len(modulated)
    spb = _samples_per_bit(n, len(bits_orig))
    analytic = hilbert(modulated)
    envelope = np.abs(analytic)
    # Downsample: take mean of each bit window
    n_bits = len(bits_orig)
    recovered = np.array([
        1 if np.mean(envelope[i*spb : (i+1)*spb]) > 0.5 else 0
        for i in range(n_bits)
    ])
    return recovered


# ── 3.2 Frequency Shift Keying (FSK) ─────────────────────────────────────────

def fsk_modulate(bits, n_samples, f0=0.03, f1=0.07, fs=FS):
    """
    Binary FSK:
      bit=0 → cosine at frequency f0
      bit=1 → cosine at frequency f1
    """
    spb = _samples_per_bit(n_samples, len(bits))
    t   = time_axis(n_samples, fs)
    out = np.zeros(n_samples)
    for i, b in enumerate(bits):
        start = i * spb
        end   = min(start + spb, n_samples)
        freq  = f1 if b == 1 else f0
        out[start:end] = np.cos(2 * np.pi * freq * t[start:end])
    return out


def fsk_demodulate(modulated, bits_orig, f0=0.03, f1=0.07, fs=FS):
    """
    FSK demodulation via matched filter (correlate each window with f0 and f1 tones).
    Decides bit based on which correlation is larger.
    """
    n   = len(modulated)
    spb = _samples_per_bit(n, len(bits_orig))
    n_bits = len(bits_orig)
    t   = time_axis(n, fs)
    ref0 = np.cos(2 * np.pi * f0 * t)
    ref1 = np.cos(2 * np.pi * f1 * t)
    recovered = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        s, e = i * spb, min((i+1) * spb, n)
        seg  = modulated[s:e]
        c0   = np.abs(np.dot(seg, ref0[s:e]))
        c1   = np.abs(np.dot(seg, ref1[s:e]))
        recovered[i] = 1 if c1 > c0 else 0
    return recovered


# ── 3.3 Phase Shift Keying (BPSK) ────────────────────────────────────────────

def psk_modulate(bits, n_samples, fc=FC, fs=FS):
    """
    Binary PSK (BPSK):
      bit=1 → cos(2π·fc·t)        (phase 0)
      bit=0 → cos(2π·fc·t + π)    (phase π, i.e. inverted carrier)
    """
    spb  = _samples_per_bit(n_samples, len(bits))
    t    = time_axis(n_samples, fs)
    base = np.cos(2 * np.pi * fc * t)
    phase_seq = np.repeat(np.where(bits == 1, 0, np.pi), spb)[:n_samples]
    if len(phase_seq) < n_samples:
        phase_seq = np.pad(phase_seq, (0, n_samples - len(phase_seq)), constant_values=phase_seq[-1])
    return np.cos(2 * np.pi * fc * t + phase_seq)


def psk_demodulate(modulated, bits_orig, fc=FC, fs=FS):
    """
    BPSK demodulation via coherent detection (multiply by reference carrier, integrate).
    Positive correlation → bit=1, negative → bit=0.
    """
    n   = len(modulated)
    spb = _samples_per_bit(n, len(bits_orig))
    t   = time_axis(n, fs)
    ref = np.cos(2 * np.pi * fc * t)
    product = modulated * ref
    n_bits = len(bits_orig)
    recovered = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        s, e = i * spb, min((i+1) * spb, n)
        recovered[i] = 1 if np.mean(product[s:e]) > 0 else 0
    return recovered


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHANNEL SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_channel(modulated, noise_std=NOISE_STD):
    """
    AWGN channel model.
    Adds Gaussian noise to represent wireless transmission impairments.
    SNR can be computed via snr_db(modulated, received).
    """
    return awgn(modulated, noise_std)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PER-STATION PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_station(station):
    """
    Full modulation pipeline for one station.

    For each signal column:
      1. Load the FIR-filtered time-series (baseband signal)
      2. Apply AM, FM, ASK, FSK, PSK modulation
      3. Pass through AWGN channel
      4. Demodulate and compute SNR / BER
      5. Save waveform plot

    Returns a list of result dicts (one per signal column per scheme).
    """
    filtered_path = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    if not os.path.exists(filtered_path):
        print(f"  [SKIP] {station}_filtered.csv not found — skipping")
        return []

    df = pd.read_csv(filtered_path, parse_dates=["timestamp"])
    sig_cols = [c for c in SIGNAL_COLS.get(station, []) if c in df.columns]
    if not sig_cols:
        print(f"  [SKIP] No matching signal columns for {station}")
        return []

    results = []

    for col in sig_cols:
        baseband = df[col].fillna(0).to_numpy(dtype=float)
        if len(baseband) < 30:
            continue

        n      = len(baseband)
        bits   = signal_to_bits(normalise(baseband))
        n_bits = len(bits)

        # ── Modulate ─────────────────────────────────────────────────────────
        am_mod  = am_modulate(baseband)
        fm_mod  = fm_modulate(baseband)
        ask_mod = ask_modulate(bits, n)
        fsk_mod = fsk_modulate(bits, n)
        psk_mod = psk_modulate(bits, n)

        # ── Channel (AWGN) ───────────────────────────────────────────────────
        am_rx  = simulate_channel(am_mod)
        fm_rx  = simulate_channel(fm_mod)
        ask_rx = simulate_channel(ask_mod)
        fsk_rx = simulate_channel(fsk_mod)
        psk_rx = simulate_channel(psk_mod)

        # ── Demodulate ───────────────────────────────────────────────────────
        am_demod  = am_demodulate(am_rx)
        fm_demod  = fm_demodulate(fm_rx)
        ask_bits  = ask_demodulate(ask_rx, bits)
        fsk_bits  = fsk_demodulate(fsk_rx, bits)
        psk_bits  = psk_demodulate(psk_rx, bits)

        # ── Metrics ──────────────────────────────────────────────────────────
        am_snr  = snr_db(am_mod, am_rx)
        fm_snr  = snr_db(fm_mod, fm_rx)
        ask_snr = snr_db(ask_mod, ask_rx)
        fsk_snr = snr_db(fsk_mod, fsk_rx)
        psk_snr = snr_db(psk_mod, psk_rx)

        ask_ber = ber(bits, ask_bits)
        fsk_ber = ber(bits, fsk_bits)
        psk_ber = ber(bits, psk_bits)

        results.append({
            "station"   : station,
            "signal_col": col,
            "AM_SNR_dB" : round(am_snr,  3),
            "FM_SNR_dB" : round(fm_snr,  3),
            "ASK_SNR_dB": round(ask_snr, 3),
            "FSK_SNR_dB": round(fsk_snr, 3),
            "PSK_SNR_dB": round(psk_snr, 3),
            "ASK_BER"   : round(ask_ber, 5),
            "FSK_BER"   : round(fsk_ber, 5),
            "PSK_BER"   : round(psk_ber, 5),
        })

        # ── Waveform Plot ─────────────────────────────────────────────────────
        _plot_waveforms(
            station, col, baseband,
            am_mod, am_rx, am_demod,
            fm_mod, fm_rx, fm_demod,
            ask_mod, ask_rx,
            fsk_mod, fsk_rx,
            psk_mod, psk_rx,
            bits, ask_bits, fsk_bits, psk_bits
        )

        print(f"  {station} · {col} | AM SNR={am_snr:.1f}dB | "
              f"ASK BER={ask_ber:.4f} | FSK BER={fsk_ber:.4f} | PSK BER={psk_ber:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

BG   = "#0d1117"
TEXT = "#e6edf3"
COLS = {
    "baseband" : "#58a6ff",
    "AM"       : "#3fb950",
    "FM"       : "#f0883e",
    "ASK"      : "#bc8cff",
    "FSK"      : "#ff7b72",
    "PSK"      : "#ffa657",
    "rx"       : "#8b949e",
    "demod"    : "#56d364",
}


def _plot_waveforms(station, col,
                    baseband,
                    am_mod, am_rx, am_demod,
                    fm_mod, fm_rx, fm_demod,
                    ask_mod, ask_rx,
                    fsk_mod, fsk_rx,
                    psk_mod, psk_rx,
                    bits, ask_bits, fsk_bits, psk_bits):
    """
    Generate a 5-row diagnostic waveform plot for one signal column.
    Rows: baseband | AM | FM | ASK | FSK | PSK
    Each row shows: modulated (clean), received (noisy), demodulated/recovered.
    """
    # Use only the first 200 samples for readability
    N   = min(200, len(baseband))
    t   = np.arange(N)
    n_b = min(N, len(bits))

    fig, axes = plt.subplots(6, 1, figsize=(16, 18), facecolor=BG)
    fig.suptitle(
        f"Modulation Waveforms — {station} · {col}\n"
        f"carrier fc={FC} Hz | AWGN σ={NOISE_STD}",
        fontsize=14, fontweight="bold", color=TEXT, y=0.995
    )

    def style_ax(ax, title, ylabel="Amplitude"):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=6)
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # Row 0 — Baseband
    axes[0].plot(t, normalise(baseband[:N]), color=COLS["baseband"], lw=1.2, label="baseband (normalised)")
    axes[0].step(t[:n_b], bits[:n_b], color="#ffa657", lw=0.8, alpha=0.5, label="bits (threshold=0.5)")
    axes[0].legend(fontsize=7, facecolor="#1c2128", labelcolor=TEXT)
    style_ax(axes[0], "Baseband Signal (FIR-filtered, normalised)")

    # Row 1 — AM
    axes[1].plot(t, am_mod[:N],   color=COLS["AM"],    lw=0.9, alpha=0.9, label="AM modulated")
    axes[1].plot(t, am_rx[:N],    color=COLS["rx"],    lw=0.7, alpha=0.5, label="received (AWGN)")
    axes[1].plot(t, am_demod[:N], color=COLS["demod"], lw=1.0, alpha=0.8, label="AM demodulated (envelope)")
    axes[1].legend(fontsize=7, facecolor="#1c2128", labelcolor=TEXT)
    style_ax(axes[1], "AM — Amplitude Modulation (index=0.8)")

    # Row 2 — FM
    axes[2].plot(t, fm_mod[:N],   color=COLS["FM"],    lw=0.9, alpha=0.9, label="FM modulated")
    axes[2].plot(t, fm_rx[:N],    color=COLS["rx"],    lw=0.7, alpha=0.5, label="received (AWGN)")
    axes[2].plot(t, fm_demod[:N], color=COLS["demod"], lw=1.0, alpha=0.8, label="FM demodulated (inst. freq.)")
    axes[2].legend(fontsize=7, facecolor="#1c2128", labelcolor=TEXT)
    style_ax(axes[2], "FM — Frequency Modulation (kf=0.1)")

    # Row 3 — ASK
    spb_ask = max(1, N // n_b)
    ask_rec_ext = np.repeat(ask_bits[:n_b], spb_ask)[:N]
    axes[3].plot(t, ask_mod[:N],    color=COLS["ASK"],   lw=0.9, alpha=0.9, label="ASK modulated (OOK)")
    axes[3].plot(t, ask_rx[:N],     color=COLS["rx"],    lw=0.7, alpha=0.5, label="received (AWGN)")
    axes[3].step(t, ask_rec_ext,    color=COLS["demod"], lw=1.2, alpha=0.9, label="ASK demodulated bits")
    axes[3].legend(fontsize=7, facecolor="#1c2128", labelcolor=TEXT)
    style_ax(axes[3], "ASK — Amplitude Shift Keying")

    # Row 4 — FSK
    fsk_rec_ext = np.repeat(fsk_bits[:n_b], spb_ask)[:N]
    axes[4].plot(t, fsk_mod[:N],    color=COLS["FSK"],   lw=0.9, alpha=0.9, label="FSK modulated (f0=0.03Hz, f1=0.07Hz)")
    axes[4].plot(t, fsk_rx[:N],     color=COLS["rx"],    lw=0.7, alpha=0.5, label="received (AWGN)")
    axes[4].step(t, fsk_rec_ext,    color=COLS["demod"], lw=1.2, alpha=0.9, label="FSK demodulated bits")
    axes[4].legend(fontsize=7, facecolor="#1c2128", labelcolor=TEXT)
    style_ax(axes[4], "FSK — Frequency Shift Keying")

    # Row 5 — PSK
    psk_rec_ext = np.repeat(psk_bits[:n_b], spb_ask)[:N]
    axes[5].plot(t, psk_mod[:N],    color=COLS["PSK"],   lw=0.9, alpha=0.9, label="PSK modulated (BPSK)")
    axes[5].plot(t, psk_rx[:N],     color=COLS["rx"],    lw=0.7, alpha=0.5, label="received (AWGN)")
    axes[5].step(t, psk_rec_ext,    color=COLS["demod"], lw=1.2, alpha=0.9, label="PSK demodulated bits")
    axes[5].legend(fontsize=7, facecolor="#1c2128", labelcolor=TEXT)
    axes[5].set_xlabel("Sample index", color="#8b949e", fontsize=8)
    style_ax(axes[5], "PSK — Binary Phase Shift Keying (BPSK)")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fname = os.path.join(OUT_DIR, f"{station}_{col}_modulation_plots.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    → Plot saved: {fname}")


def plot_ber_comparison(all_results):
    """
    Cross-station BER and SNR comparison bar chart.
    One figure with subplots for BER (ASK/FSK/PSK) and SNR (AM/FM/ASK/FSK/PSK).
    """
    if not all_results:
        return

    df = pd.DataFrame(all_results)
    # Aggregate: mean per station
    agg = df.groupby("station")[
        ["AM_SNR_dB","FM_SNR_dB","ASK_SNR_dB","FSK_SNR_dB","PSK_SNR_dB",
         "ASK_BER","FSK_BER","PSK_BER"]
    ].mean().reset_index()

    stations = agg["station"].tolist()
    x = np.arange(len(stations))
    w = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.suptitle(
        "Modulation Performance Comparison — All Stations\nTELE 523 · Group 1",
        fontsize=15, fontweight="bold", color=TEXT
    )

    # ── SNR plot ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    schemes_snr = ["AM_SNR_dB","FM_SNR_dB","ASK_SNR_dB","FSK_SNR_dB","PSK_SNR_dB"]
    labels_snr  = ["AM","FM","ASK","FSK","PSK"]
    pal         = [COLS["AM"], COLS["FM"], COLS["ASK"], COLS["FSK"], COLS["PSK"]]
    for i, (col, lbl, c) in enumerate(zip(schemes_snr, labels_snr, pal)):
        bars = ax1.bar(x + i*w, agg[col], w, label=lbl, color=c, alpha=0.85, edgecolor="#ffffff11")
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}",
                     ha="center", va="bottom", color=TEXT, fontsize=6.5)
    ax1.set_xticks(x + 2*w)
    ax1.set_xticklabels(stations, color=TEXT, fontsize=9)
    ax1.set_ylabel("SNR (dB)", color="#8b949e")
    ax1.set_title("Signal-to-Noise Ratio per Station & Scheme", color=TEXT, fontsize=11)
    ax1.legend(facecolor="#1c2128", labelcolor=TEXT, fontsize=9)
    ax1.tick_params(colors="#8b949e")
    for sp in ax1.spines.values(): sp.set_color("#30363d")

    # ── BER plot ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    schemes_ber = ["ASK_BER","FSK_BER","PSK_BER"]
    labels_ber  = ["ASK","FSK","PSK"]
    pal_ber     = [COLS["ASK"], COLS["FSK"], COLS["PSK"]]
    w2 = 0.25
    for i, (col, lbl, c) in enumerate(zip(schemes_ber, labels_ber, pal_ber)):
        bars = ax2.bar(x + i*w2, agg[col], w2, label=lbl, color=c, alpha=0.85, edgecolor="#ffffff11")
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x()+bar.get_width()/2, h+0.0005, f"{h:.4f}",
                     ha="center", va="bottom", color=TEXT, fontsize=6.5)
    ax2.set_xticks(x + w2)
    ax2.set_xticklabels(stations, color=TEXT, fontsize=9)
    ax2.set_ylabel("Bit Error Rate (BER)", color="#8b949e")
    ax2.set_title("BER per Station & Digital Scheme", color=TEXT, fontsize=11)
    ax2.legend(facecolor="#1c2128", labelcolor=TEXT, fontsize=9)
    ax2.tick_params(colors="#8b949e")
    for sp in ax2.spines.values(): sp.set_color("#30363d")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "modulation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nSummary plot saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("TELE 523 · Modulation Lead — Student 3")
    print("Industrial Machine Condition Monitoring")
    print("=" * 65)
    print(f"\nCarrier frequency : {FC} Hz")
    print(f"Sampling rate     : {FS} Hz")
    print(f"Nyquist limit     : {FS/2} Hz")
    print(f"AWGN noise σ      : {NOISE_STD}")
    print(f"Output directory  : {OUT_DIR}\n")

    all_results = []

    for station in sorted(SIGNAL_COLS.keys()):
        print(f"\n── Processing {station} ─────────────────────────────────────")
        results = process_station(station)
        all_results.extend(results)

    if not all_results:
        print("\n[ERROR] No results — check that data/processed/ contains *_filtered.csv files.")
        return

    # ── Save metrics CSV ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_out = os.path.join(OUT_DIR, "modulation_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"\nResults CSV saved → {csv_out}")

    # ── Summary comparison plot ───────────────────────────────────────────────
    plot_ber_comparison(all_results)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MODULATION RESULTS SUMMARY")
    print("=" * 65)
    summary = results_df.groupby("station")[
        ["AM_SNR_dB","FM_SNR_dB","ASK_SNR_dB","FSK_SNR_dB","PSK_SNR_dB",
         "ASK_BER","FSK_BER","PSK_BER"]
    ].mean().round(4)
    print(summary.to_string())
    print("\nPhase 3 — Modulation complete.")


if __name__ == "__main__":
    main()