"""
modulation.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Core modulation engine. Every station's signal columns pass through each of
the 5 schemes (AM, FM, ASK, FSK, PSK) independently.

Inputs  (data/processed/):   {STATION}_filtered.csv
Outputs (results/modulation/):
  Plots   — {STATION}_{SCHEME}_{col}_plot.png   (one per scheme per signal col)
  Metrics — modulation_results.csv             (SNR + BER for every combination)
  Summary — modulation_summary.png             (cross-station comparison chart)

Handoff CSVs for Student 4 are written by generate_modulation_outputs.py,
which imports the modulation functions directly from this file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR        = os.path.join(BASE_DIR, "results", "modulation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared parameters ─────────────────────────────────────────────────────────
FS        = 0.5    # Hz  — sampling rate (inherited from Student 2)
FC        = 0.05   # Hz  — carrier frequency (below Nyquist = 0.25 Hz)
NOISE_STD = 0.05   # sigma — AWGN channel noise standard deviation
MOD_INDEX = 0.8    # AM modulation index
KF        = 0.1    # FM frequency deviation constant (Hz per unit amplitude)
F0_FSK    = 0.03   # Hz  — FSK frequency for bit = 0
F1_FSK    = 0.07   # Hz  — FSK frequency for bit = 1

# Signal columns to process per station (chosen by Student 2 as the baseband)
SIGNAL_COLS = {
    "EC_1"  : ["i3_photoresistor", "current_task_duration"],
    "HBW_1" : ["m1_speed", "m2_speed", "m3_speed", "m4_speed", "current_task_duration"],
    "MM_1"  : ["m1_speed", "m2_speed", "m3_speed", "current_task_duration"],
    "OV_1"  : ["m1_speed", "current_task_duration"],
    "SM_1"  : ["m1_speed", "current_task_duration"],
    "VGR_1" : ["m1_speed", "m2_speed", "m3_speed", "current_task_duration"],
    "WT_1"  : ["m2_speed", "current_task_duration"],
}


# =============================================================================
# SECTION 1 — UTILITIES
# =============================================================================

def time_axis(n, fs=FS):
    """Return a uniformly-spaced time array of n samples at rate fs (Hz)."""
    return np.arange(n) / fs


def normalise(signal):
    """Min-max normalise a signal to [0, 1]. Returns zeros for constant signals."""
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-12:
        return np.zeros_like(signal, dtype=float)
    return (signal - lo) / (hi - lo)


def awgn(signal, noise_std=NOISE_STD, seed=42):
    """Add Additive White Gaussian Noise — models the wireless channel."""
    rng = np.random.default_rng(seed)
    return signal + rng.normal(0, noise_std, size=len(signal))


def signal_to_bits(signal):
    """Convert a normalised [0,1] signal to binary by thresholding at 0.5."""
    return (np.clip(normalise(signal), 0, 1) > 0.5).astype(int)


def _spb(n_samples, n_bits):
    """Samples per bit — integer floor division, minimum 1."""
    return max(1, n_samples // max(1, n_bits))


def snr_db(clean, noisy):
    """Signal-to-Noise Ratio in dB between a clean and a noisy version."""
    sp  = np.mean(clean ** 2)
    np_ = np.mean((clean - noisy) ** 2)
    if np_ == 0: return float("inf")
    if sp  == 0: return float("-inf")
    return 10 * np.log10(sp / np_)


def ber(original_bits, recovered_bits):
    """Bit Error Rate — fraction of bits that differ after recovery."""
    n = min(len(original_bits), len(recovered_bits))
    return float(np.sum(original_bits[:n] != recovered_bits[:n]) / n)


def _expand_bits(bits, n_samples):
    """Expand a bit array back to sample resolution (repeat each bit _spb times)."""
    s   = _spb(n_samples, len(bits))
    out = np.repeat(bits.astype(float), s)[:n_samples]
    if len(out) < n_samples:
        out = np.pad(out, (0, n_samples - len(out)), constant_values=out[-1])
    return out


# =============================================================================
# SECTION 2 — AM  (Amplitude Modulation)
# =============================================================================

def am_modulate(baseband, fc=FC, fs=FS, mod_index=MOD_INDEX):
    """
    AM modulation:  s(t) = [1 + m*x(t)] * cos(2*pi*fc*t)
    x(t) is the normalised baseband signal; m is the modulation index.
    """
    x = normalise(baseband)
    t = time_axis(len(x), fs)
    return (1 + mod_index * x) * np.cos(2 * np.pi * fc * t)


def am_demodulate(received, fs=FS):
    """
    AM demodulation via Hilbert envelope detection.
    Returns the envelope with DC offset removed, normalised to [0, 1].
    """
    env = np.abs(hilbert(received))
    env -= env.mean()
    return np.clip(normalise(env), 0, 1)


# =============================================================================
# SECTION 3 — FM  (Frequency Modulation)
# =============================================================================

def fm_modulate(baseband, fc=FC, fs=FS, kf=KF):
    """
    FM modulation:  s(t) = cos(2*pi*fc*t + 2*pi*kf * integral(x))
    kf is the frequency deviation constant.
    Carson's rule bandwidth = 2*(kf*A + W).
    """
    x     = normalise(baseband)
    t     = time_axis(len(x), fs)
    integ = np.cumsum(x) / fs
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integ)


def fm_demodulate(received, fc=FC, fs=FS):
    """
    FM demodulation via instantaneous frequency (phase-differentiator).
    Returns the recovered signal normalised to [0, 1].
    """
    phase     = np.unwrap(np.angle(hilbert(received)))
    inst_freq = np.diff(phase) / (2 * np.pi / fs)
    inst_freq = np.append(inst_freq, inst_freq[-1]) - fc
    return np.clip(normalise(inst_freq), 0, 1)


# =============================================================================
# SECTION 4 — ASK  (Amplitude Shift Keying / OOK)
# =============================================================================

def ask_modulate(bits, n_samples, fc=FC, fs=FS):
    """
    Binary ASK (On-Off Keying):
      bit = 1 -> full-amplitude carrier
      bit = 0 -> carrier off (zero)
    """
    t   = time_axis(n_samples, fs)
    env = _expand_bits(bits, n_samples)
    return env * np.cos(2 * np.pi * fc * t)


def ask_demodulate(received, bits_orig, fs=FS):
    """
    ASK demodulation via envelope detection + per-window threshold.
    Returns:
      recovered_bits — 0/1 array at bit resolution
      envelope       — continuous envelope at sample resolution, normalised [0,1]
    """
    n   = len(received)
    nb  = len(bits_orig)
    s   = _spb(n, nb)
    env = np.abs(hilbert(received))
    recovered = np.array([
        1 if np.mean(env[i*s : (i+1)*s]) > 0.5 else 0
        for i in range(nb)
    ])
    return recovered, np.clip(normalise(env), 0, 1)


# =============================================================================
# SECTION 5 — FSK  (Frequency Shift Keying)
# =============================================================================

def fsk_modulate(bits, n_samples, f0=F0_FSK, f1=F1_FSK, fs=FS):
    """
    Binary FSK:
      bit = 0 -> cosine at f0
      bit = 1 -> cosine at f1
    """
    s   = _spb(n_samples, len(bits))
    t   = time_axis(n_samples, fs)
    out = np.zeros(n_samples)
    for i, b in enumerate(bits):
        st, en = i*s, min((i+1)*s, n_samples)
        out[st:en] = np.cos(2 * np.pi * (f1 if b else f0) * t[st:en])
    return out


def fsk_demodulate(received, bits_orig, f0=F0_FSK, f1=F1_FSK, fs=FS):
    """
    FSK demodulation via matched filter correlation.
    Each bit window correlated with f0 and f1 reference tones;
    larger absolute dot-product wins.
    """
    n   = len(received)
    nb  = len(bits_orig)
    s   = _spb(n, nb)
    t   = time_axis(n, fs)
    r0  = np.cos(2 * np.pi * f0 * t)
    r1  = np.cos(2 * np.pi * f1 * t)
    recovered = np.zeros(nb, dtype=int)
    for i in range(nb):
        st, en = i*s, min((i+1)*s, n)
        recovered[i] = 1 if abs(np.dot(received[st:en], r1[st:en])) > \
                             abs(np.dot(received[st:en], r0[st:en])) else 0
    return recovered


# =============================================================================
# SECTION 6 — PSK  (Binary Phase Shift Keying)
# =============================================================================

def psk_modulate(bits, n_samples, fc=FC, fs=FS):
    """
    BPSK:
      bit = 1 -> cos(2*pi*fc*t)      (phase = 0)
      bit = 0 -> cos(2*pi*fc*t + pi) (phase = pi, inverted carrier)
    """
    t         = time_axis(n_samples, fs)
    phase_seq = _expand_bits(np.where(bits == 1, 0.0, np.pi), n_samples)
    return np.cos(2 * np.pi * fc * t + phase_seq)


def psk_demodulate(received, bits_orig, fc=FC, fs=FS):
    """
    BPSK coherent demodulation: multiply by reference carrier, integrate per
    bit window. Positive mean -> bit=1, negative -> bit=0.
    """
    n   = len(received)
    nb  = len(bits_orig)
    s   = _spb(n, nb)
    t   = time_axis(n, fs)
    ref = np.cos(2 * np.pi * fc * t)
    product = received * ref
    recovered = np.zeros(nb, dtype=int)
    for i in range(nb):
        st, en = i*s, min((i+1)*s, n)
        recovered[i] = 1 if np.mean(product[st:en]) > 0 else 0
    return recovered


# =============================================================================
# SECTION 7 — CHANNEL
# =============================================================================

def channel(modulated, noise_std=NOISE_STD):
    """AWGN channel — adds Gaussian noise to simulate wireless transmission."""
    return awgn(modulated, noise_std)


# =============================================================================
# SECTION 8 — PER-SCHEME PIPELINE RUNNERS
# Each function runs one complete scheme independently for one signal column.
# These are also imported by generate_modulation_outputs.py.
# =============================================================================

def run_am(baseband):
    """
    Full AM pipeline for one signal column.
    Returns: (modulated, received, demodulated, snr_dB)
    """
    mod = am_modulate(baseband)
    rx  = channel(mod)
    dem = am_demodulate(rx)
    return mod, rx, dem, snr_db(mod, rx)


def run_fm(baseband):
    """
    Full FM pipeline for one signal column.
    Returns: (modulated, received, demodulated, snr_dB)
    """
    mod = fm_modulate(baseband)
    rx  = channel(mod)
    dem = fm_demodulate(rx)
    return mod, rx, dem, snr_db(mod, rx)


def run_ask(baseband):
    """
    Full ASK pipeline for one signal column.
    Returns: (modulated, received, recovered_bits, envelope, snr_dB, ber_val, original_bits)
    """
    bits     = signal_to_bits(baseband)
    mod      = ask_modulate(bits, len(baseband))
    rx       = channel(mod)
    rec, env = ask_demodulate(rx, bits)
    return mod, rx, rec, env, snr_db(mod, rx), ber(bits, rec), bits


def run_fsk(baseband):
    """
    Full FSK pipeline for one signal column.
    Returns: (modulated, received, recovered_bits, snr_dB, ber_val, original_bits)
    """
    bits = signal_to_bits(baseband)
    mod  = fsk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = fsk_demodulate(rx, bits)
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits


def run_psk(baseband):
    """
    Full PSK pipeline for one signal column.
    Returns: (modulated, received, recovered_bits, snr_dB, ber_val, original_bits)
    """
    bits = signal_to_bits(baseband)
    mod  = psk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = psk_demodulate(rx, bits)
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits


# =============================================================================
# SECTION 9 — STATION PROCESSING
# Outer loop = schemes, inner loop = signal columns.
# This makes it explicit that every column passes through every scheme
# as a completely independent pipeline run.
# =============================================================================

def process_station(station):
    """
    For one station, run every signal column through every scheme independently.

    Loop order:  for each scheme -> for each signal column
      modulate -> channel -> demodulate -> compute SNR/BER -> save plot

    Returns a list of metric dicts (one per scheme x signal column combination).
    """
    fpath = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    if not os.path.exists(fpath):
        print(f"  [SKIP] {station}_filtered.csv not found")
        return []

    df       = pd.read_csv(fpath, parse_dates=["timestamp"])
    sig_cols = [c for c in SIGNAL_COLS.get(station, []) if c in df.columns]
    if not sig_cols:
        print(f"  [SKIP] No matching signal columns for {station}")
        return []

    results = []

    # Outer loop over schemes — makes independence explicit
    for scheme in ["AM", "FM", "ASK", "FSK", "PSK"]:
        print(f"  [{scheme}]")
        for col in sig_cols:
            baseband = df[col].fillna(0).to_numpy(dtype=float)
            if len(baseband) < 30:
                continue

            row = {"station": station, "signal_col": col, "scheme": scheme}

            if scheme == "AM":
                mod, rx, dem, snr = run_am(baseband)
                row["SNR_dB"] = round(snr, 3)
                row["BER"]    = None
                _plot_scheme(station, col, "AM", baseband, mod, rx, dem,
                             label_mod="AM modulated",
                             label_demod="AM demodulated (envelope)")

            elif scheme == "FM":
                mod, rx, dem, snr = run_fm(baseband)
                row["SNR_dB"] = round(snr, 3)
                row["BER"]    = None
                _plot_scheme(station, col, "FM", baseband, mod, rx, dem,
                             label_mod="FM modulated",
                             label_demod="FM demodulated (inst. freq.)")

            elif scheme == "ASK":
                mod, rx, rec, env, snr, ber_val, orig_bits = run_ask(baseband)
                dem = _expand_bits(rec, len(baseband))
                row["SNR_dB"] = round(snr, 3)
                row["BER"]    = round(ber_val, 5)
                _plot_scheme(station, col, "ASK", baseband, mod, rx, dem,
                             label_mod="ASK modulated (OOK)",
                             label_demod="ASK recovered bits")

            elif scheme == "FSK":
                mod, rx, rec, snr, ber_val, orig_bits = run_fsk(baseband)
                dem = _expand_bits(rec, len(baseband))
                row["SNR_dB"] = round(snr, 3)
                row["BER"]    = round(ber_val, 5)
                _plot_scheme(station, col, "FSK", baseband, mod, rx, dem,
                             label_mod=f"FSK modulated (f0={F0_FSK}Hz, f1={F1_FSK}Hz)",
                             label_demod="FSK recovered bits")

            elif scheme == "PSK":
                mod, rx, rec, snr, ber_val, orig_bits = run_psk(baseband)
                dem = _expand_bits(rec, len(baseband))
                row["SNR_dB"] = round(snr, 3)
                row["BER"]    = round(ber_val, 5)
                _plot_scheme(station, col, "PSK", baseband, mod, rx, dem,
                             label_mod="PSK modulated (BPSK)",
                             label_demod="PSK recovered bits")

            results.append(row)
            ber_str = f"BER={row['BER']:.5f}" if row["BER"] is not None else "BER=N/A (analog)"
            print(f"      {col:30s} SNR={row['SNR_dB']:6.1f} dB | {ber_str}")

    return results


# =============================================================================
# SECTION 10 — PLOTTING
# =============================================================================

_BG      = "#0d1117"
_TEXT    = "#e6edf3"
_PALETTE = {
    "baseband" : "#58a6ff",
    "modulated": "#3fb950",
    "received" : "#8b949e",
    "demod"    : "#ffa657",
}


def _plot_scheme(station, col, scheme, baseband,
                 mod, rx, dem,
                 label_mod="modulated",
                 label_demod="demodulated"):
    """
    3-row diagnostic plot for one scheme x one signal column.
    Row 1: normalised baseband
    Row 2: modulated (clean) vs received (noisy after channel)
    Row 3: demodulated / recovered output
    """
    N = min(200, len(baseband))
    t = np.arange(N)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), facecolor=_BG)
    fig.suptitle(
        f"{station} · {col} — {scheme} Modulation\n"
        f"fc={FC} Hz | AWGN sigma={NOISE_STD} | fs={FS} Hz",
        fontsize=12, fontweight="bold", color=_TEXT, y=1.00
    )

    def style(ax, title):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color=_TEXT, fontsize=9, loc="left", pad=4)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.set_ylabel("Amplitude", color="#8b949e", fontsize=8)

    axes[0].plot(t, normalise(baseband[:N]),
                 color=_PALETTE["baseband"], lw=1.2, label="baseband (normalised)")
    axes[0].legend(fontsize=7, facecolor="#1c2128", labelcolor=_TEXT)
    style(axes[0], "Baseband signal (FIR-filtered input from Student 2)")

    axes[1].plot(t, mod[:N], color=_PALETTE["modulated"], lw=0.9, alpha=0.9, label=label_mod)
    axes[1].plot(t, rx[:N],  color=_PALETTE["received"],  lw=0.7, alpha=0.5,
                 label="received after AWGN channel")
    axes[1].legend(fontsize=7, facecolor="#1c2128", labelcolor=_TEXT)
    style(axes[1], f"{scheme} — Modulated & Channel Output")

    axes[2].plot(t, dem[:N], color=_PALETTE["demod"], lw=1.1, label=label_demod)
    axes[2].legend(fontsize=7, facecolor="#1c2128", labelcolor=_TEXT)
    axes[2].set_xlabel("Sample index", color="#8b949e", fontsize=8)
    style(axes[2], f"{scheme} — Demodulated / Recovered Output (handoff to Student 4)")

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{station}_{scheme}_{col}_plot.png")
    plt.savefig(fname, dpi=140, bbox_inches="tight", facecolor=_BG)
    plt.close()


def plot_summary(all_results):
    """Cross-station SNR and BER comparison bar chart."""
    if not all_results:
        return

    df       = pd.DataFrame(all_results)
    stations = sorted(df["station"].unique())
    schemes  = ["AM", "FM", "ASK", "FSK", "PSK"]
    x = np.arange(len(stations))
    w = 0.15
    colors = ["#3fb950", "#f0883e", "#bc8cff", "#ff7b72", "#ffa657"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=_BG)
    fig.suptitle("Modulation Performance Summary — All Stations\nTELE 523 · Group 1",
                 fontsize=14, fontweight="bold", color=_TEXT)

    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    for i, (s, c) in enumerate(zip(schemes, colors)):
        vals = [df[(df.station == st) & (df.scheme == s)]["SNR_dB"].mean()
                for st in stations]
        bars = ax1.bar(x + i*w, vals, w, label=s, color=c, alpha=0.85,
                       edgecolor="#ffffff11")
        for b in bars:
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                     f"{b.get_height():.1f}", ha="center", va="bottom",
                     color=_TEXT, fontsize=6)
    ax1.set_xticks(x + 2*w)
    ax1.set_xticklabels(stations, color=_TEXT, fontsize=9)
    ax1.set_ylabel("SNR (dB)", color="#8b949e")
    ax1.set_title("Signal-to-Noise Ratio per Station & Scheme", color=_TEXT, fontsize=11)
    ax1.legend(facecolor="#1c2128", labelcolor=_TEXT, fontsize=9)
    ax1.tick_params(colors="#8b949e")
    for sp in ax1.spines.values():
        sp.set_color("#30363d")

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    dig_schemes = ["ASK", "FSK", "PSK"]
    dig_colors  = ["#bc8cff", "#ff7b72", "#ffa657"]
    w2 = 0.25
    for i, (s, c) in enumerate(zip(dig_schemes, dig_colors)):
        vals = [df[(df.station == st) & (df.scheme == s)]["BER"].mean()
                for st in stations]
        bars = ax2.bar(x + i*w2, vals, w2, label=s, color=c, alpha=0.85,
                       edgecolor="#ffffff11")
        for b in bars:
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
                     f"{b.get_height():.4f}", ha="center", va="bottom",
                     color=_TEXT, fontsize=6)
    ax2.set_xticks(x + w2)
    ax2.set_xticklabels(stations, color=_TEXT, fontsize=9)
    ax2.set_ylabel("Bit Error Rate (BER)", color="#8b949e")
    ax2.set_title("BER per Station & Digital Scheme", color=_TEXT, fontsize=11)
    ax2.legend(facecolor="#1c2128", labelcolor=_TEXT, fontsize=9)
    ax2.tick_params(colors="#8b949e")
    for sp in ax2.spines.values():
        sp.set_color("#30363d")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "modulation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"Summary plot saved -> {out}")


# =============================================================================
# SECTION 11 — MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TELE 523 - Modulation Lead (Student 3)")
    print("Each station x each signal column x each scheme — independently")
    print("=" * 70)
    print(f"  fc={FC} Hz | fs={FS} Hz | Nyquist={FS/2} Hz | AWGN sigma={NOISE_STD}\n")

    all_results = []

    for station in sorted(SIGNAL_COLS.keys()):
        print(f"\n-- {station} " + "-"*50)
        results = process_station(station)
        all_results.extend(results)

    if not all_results:
        print("\n[ERROR] No results — check data/processed/ for *_filtered.csv files.")
        return

    df      = pd.DataFrame(all_results)
    csv_out = os.path.join(OUT_DIR, "modulation_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nMetrics CSV saved -> {csv_out}")

    plot_summary(all_results)

    print("\n" + "=" * 70)
    print("SUMMARY — mean SNR (dB) per station x scheme")
    pivot_snr = df.pivot_table(
        values="SNR_dB", index="station", columns="scheme", aggfunc="mean"
    ).round(2)
    print(pivot_snr.to_string())

    print("\nBER (digital schemes only):")
    pivot_ber = df[df["BER"].notna()].pivot_table(
        values="BER", index="station", columns="scheme", aggfunc="mean"
    ).round(5)
    print(pivot_ber.to_string())
    print("\nModulation phase complete.")


if __name__ == "__main__":
    main()