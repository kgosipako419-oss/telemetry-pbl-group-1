"""
modulation.py  —  Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Core modulation engine.

Inputs (data/processed/):
  {STATION}_filtered.csv   — FIR-filtered time-series from Student 2 (1893 rows)
  {STATION}_features.csv   — segment-level PSD/RMS feature vectors from Student 2 (125 rows)

Both file types are treated as baseband signal sources.
Every column in every file passes through all 5 schemes independently.

Outputs (results/modulation/):
  Diagnostic plots  —  {STATION}_{SOURCE}_{SCHEME}_{col}_plot.png
  Metrics CSV       —  modulation_results.csv
  Summary chart     —  modulation_summary.png

The run_* functions are public and imported by generate_modulation_outputs.py
and demodulation.py.
"""

import os
import sys
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

# ── Carrier / channel parameters ──────────────────────────────────────────────
FS        = 0.5    # Hz  — sampling rate inherited from Student 2
FC        = 0.05   # Hz  — carrier (< Nyquist 0.25 Hz)
NOISE_STD = 0.05   # σ   — AWGN noise standard deviation
MOD_INDEX = 0.8    # AM  — modulation index
KF        = 0.1    # FM  — frequency deviation constant
F0_FSK    = 0.03   # Hz  — FSK bit=0 frequency
F1_FSK    = 0.07   # Hz  — FSK bit=1 frequency

# ── Signal columns from _filtered.csv per station ─────────────────────────────
SIGNAL_COLS_FILTERED = {
    "EC_1"  : ["i3_photoresistor"],
    "HBW_1" : ["i4_light_barrier","m1_speed","m2_speed","m3_speed","current_task",
                "current_task_duration","current_sub_task","i1_light_barrier",
                "i2_light_barrier","i3_light_barrier","i5_pos_switch","i6_pos_switch",
                "i7_pos_switch","i8_pos_switch","m4_speed","current_pos_x",
                "current_pos_y","target_pos_x","target_pos_y","amount_of_workpieces"],
    "MM_1"  : ["i1_pos_switch","i2_pos_switch","i3_pos_switch","i4_light_barrier",
                "m1_speed","m2_speed","m3_speed","o7_valve","o8_compressor",
                "current_task","current_task_duration","current_sub_task"],
    "OV_1"  : ["i1_pos_switch","i2_pos_switch","m1_speed","o7_valve","o8_compressor",
                "current_task","current_task_duration","current_sub_task","i5_light_barrier"],
    "SM_1"  : ["m1_speed","o7_valve","o8_compressor","current_task","current_task_duration",
                "current_sub_task","i1_light_barrier","i3_light_barrier","i2_color_sensor",
                "i6_light_barrier","i7_light_barrier","i8_light_barrier","o5_valve"],
    "VGR_1" : ["i1_pos_switch","i2_pos_switch","i3_pos_switch","i4_light_barrier",
                "m1_speed","m2_speed","m3_speed","current_task","current_task_duration",
                "current_sub_task","current_pos_x","current_pos_y","target_pos_x",
                "target_pos_y","i7_light_barrier","i8_color_sensor","o7_compressor_level",
                "o8_valve_open","current_pos_z","target_pos_z"],
    "WT_1"  : ["i3_pos_switch","m2_speed","o8_compressor","o5_valve","o6_valve"],
}

# ── Signal columns from _features.csv per station ─────────────────────────────
SIGNAL_COLS_FEATURES = {
    "EC_1"  : ["i3_photoresistor_psd_peak","i3_photoresistor_psd_mean_energy",
                "i3_photoresistor_psd_peak_freq","i3_photoresistor_rms",
                "i3_photoresistor_mean","i3_photoresistor_variance",
                "i3_photoresistor_fft_peak","current_task_duration_psd_peak",
                "current_task_duration_psd_mean_energy","current_task_duration_psd_peak_freq",
                "current_task_duration_rms","current_task_duration_mean",
                "current_task_duration_variance","current_task_duration_fft_peak"],
    "HBW_1" : ["m1_speed_psd_peak","m1_speed_psd_mean_energy","m1_speed_psd_peak_freq",
                "m1_speed_rms","m1_speed_mean","m1_speed_variance","m1_speed_fft_peak",
                "m2_speed_psd_peak","m2_speed_psd_mean_energy","m2_speed_psd_peak_freq",
                "m2_speed_rms","m2_speed_mean","m2_speed_variance","m2_speed_fft_peak",
                "m3_speed_psd_peak","m3_speed_psd_mean_energy","m3_speed_psd_peak_freq",
                "m3_speed_rms","m3_speed_mean","m3_speed_variance","m3_speed_fft_peak",
                "m4_speed_psd_peak","m4_speed_psd_mean_energy","m4_speed_psd_peak_freq",
                "m4_speed_rms","m4_speed_mean","m4_speed_variance","m4_speed_fft_peak",
                "current_pos_x_psd_peak","current_pos_x_psd_mean_energy",
                "current_pos_x_psd_peak_freq","current_pos_x_rms","current_pos_x_mean",
                "current_pos_x_variance","current_pos_x_fft_peak",
                "current_pos_y_psd_peak","current_pos_y_psd_mean_energy",
                "current_pos_y_psd_peak_freq","current_pos_y_rms","current_pos_y_mean",
                "current_pos_y_variance","current_pos_y_fft_peak",
                "target_pos_x_psd_peak","target_pos_x_psd_mean_energy",
                "target_pos_x_psd_peak_freq","target_pos_x_rms","target_pos_x_mean",
                "target_pos_x_variance","target_pos_x_fft_peak",
                "target_pos_y_psd_peak","target_pos_y_psd_mean_energy",
                "target_pos_y_psd_peak_freq","target_pos_y_rms","target_pos_y_mean",
                "target_pos_y_variance","target_pos_y_fft_peak",
                "current_task_duration_psd_peak","current_task_duration_psd_mean_energy",
                "current_task_duration_psd_peak_freq","current_task_duration_rms",
                "current_task_duration_mean","current_task_duration_variance",
                "current_task_duration_fft_peak"],
    "MM_1"  : ["m1_speed_psd_peak","m1_speed_psd_mean_energy","m1_speed_psd_peak_freq",
                "m1_speed_rms","m1_speed_mean","m1_speed_variance","m1_speed_fft_peak",
                "m2_speed_psd_peak","m2_speed_psd_mean_energy","m2_speed_psd_peak_freq",
                "m2_speed_rms","m2_speed_mean","m2_speed_variance","m2_speed_fft_peak",
                "m3_speed_psd_peak","m3_speed_psd_mean_energy","m3_speed_psd_peak_freq",
                "m3_speed_rms","m3_speed_mean","m3_speed_variance","m3_speed_fft_peak",
                "o8_compressor_psd_peak","o8_compressor_psd_mean_energy",
                "o8_compressor_psd_peak_freq","o8_compressor_rms","o8_compressor_mean",
                "o8_compressor_variance","o8_compressor_fft_peak",
                "current_task_duration_psd_peak","current_task_duration_psd_mean_energy",
                "current_task_duration_psd_peak_freq","current_task_duration_rms",
                "current_task_duration_mean","current_task_duration_variance",
                "current_task_duration_fft_peak"],
    "OV_1"  : ["m1_speed_psd_peak","m1_speed_psd_mean_energy","m1_speed_psd_peak_freq",
                "m1_speed_rms","m1_speed_mean","m1_speed_variance","m1_speed_fft_peak",
                "o8_compressor_psd_peak","o8_compressor_psd_mean_energy",
                "o8_compressor_psd_peak_freq","o8_compressor_rms","o8_compressor_mean",
                "o8_compressor_variance","o8_compressor_fft_peak",
                "current_task_duration_psd_peak","current_task_duration_psd_mean_energy",
                "current_task_duration_psd_peak_freq","current_task_duration_rms",
                "current_task_duration_mean","current_task_duration_variance",
                "current_task_duration_fft_peak"],
    "SM_1"  : ["m1_speed_psd_peak","m1_speed_psd_mean_energy","m1_speed_psd_peak_freq",
                "m1_speed_rms","m1_speed_mean","m1_speed_variance","m1_speed_fft_peak",
                "o8_compressor_psd_peak","o8_compressor_psd_mean_energy",
                "o8_compressor_psd_peak_freq","o8_compressor_rms","o8_compressor_mean",
                "o8_compressor_variance","o8_compressor_fft_peak",
                "current_task_duration_psd_peak","current_task_duration_psd_mean_energy",
                "current_task_duration_psd_peak_freq","current_task_duration_rms",
                "current_task_duration_mean","current_task_duration_variance",
                "current_task_duration_fft_peak"],
    "VGR_1" : ["m1_speed_psd_peak","m1_speed_psd_mean_energy","m1_speed_psd_peak_freq",
                "m1_speed_rms","m1_speed_mean","m1_speed_variance","m1_speed_fft_peak",
                "m2_speed_psd_peak","m2_speed_psd_mean_energy","m2_speed_psd_peak_freq",
                "m2_speed_rms","m2_speed_mean","m2_speed_variance","m2_speed_fft_peak",
                "m3_speed_psd_peak","m3_speed_psd_mean_energy","m3_speed_psd_peak_freq",
                "m3_speed_rms","m3_speed_mean","m3_speed_variance","m3_speed_fft_peak",
                "current_pos_x_psd_peak","current_pos_x_psd_mean_energy",
                "current_pos_x_psd_peak_freq","current_pos_x_rms","current_pos_x_mean",
                "current_pos_x_variance","current_pos_x_fft_peak",
                "current_pos_y_psd_peak","current_pos_y_psd_mean_energy",
                "current_pos_y_psd_peak_freq","current_pos_y_rms","current_pos_y_mean",
                "current_pos_y_variance","current_pos_y_fft_peak",
                "current_pos_z_psd_peak","current_pos_z_psd_mean_energy",
                "current_pos_z_psd_peak_freq","current_pos_z_rms","current_pos_z_mean",
                "current_pos_z_variance","current_pos_z_fft_peak",
                "target_pos_x_psd_peak","target_pos_x_psd_mean_energy",
                "target_pos_x_psd_peak_freq","target_pos_x_rms","target_pos_x_mean",
                "target_pos_x_variance","target_pos_x_fft_peak",
                "target_pos_y_psd_peak","target_pos_y_psd_mean_energy",
                "target_pos_y_psd_peak_freq","target_pos_y_rms","target_pos_y_mean",
                "target_pos_y_variance","target_pos_y_fft_peak",
                "target_pos_z_psd_peak","target_pos_z_psd_mean_energy",
                "target_pos_z_psd_peak_freq","target_pos_z_rms","target_pos_z_mean",
                "target_pos_z_variance","target_pos_z_fft_peak",
                "current_task_duration_psd_peak","current_task_duration_psd_mean_energy",
                "current_task_duration_psd_peak_freq","current_task_duration_rms",
                "current_task_duration_mean","current_task_duration_variance",
                "current_task_duration_fft_peak"],
    "WT_1"  : ["m2_speed_psd_peak","m2_speed_psd_mean_energy","m2_speed_psd_peak_freq",
                "m2_speed_rms","m2_speed_mean","m2_speed_variance","m2_speed_fft_peak",
                "o8_compressor_psd_peak","o8_compressor_psd_mean_energy",
                "o8_compressor_psd_peak_freq","o8_compressor_rms","o8_compressor_mean",
                "o8_compressor_variance","o8_compressor_fft_peak",
                "current_task_duration_psd_peak","current_task_duration_psd_mean_energy",
                "current_task_duration_psd_peak_freq","current_task_duration_rms",
                "current_task_duration_mean","current_task_duration_variance",
                "current_task_duration_fft_peak"],
}

STATIONS = ["EC_1", "HBW_1", "MM_1", "OV_1", "SM_1", "VGR_1", "WT_1"]
SCHEMES  = ["AM", "FM", "ASK", "FSK", "PSK"]


# =============================================================================
# SECTION 1 — UTILITIES
# =============================================================================

def time_axis(n, fs=FS):
    return np.arange(n) / fs

def normalise(signal):
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-12:
        return np.zeros_like(signal, dtype=float)
    return (signal - lo) / (hi - lo)

def awgn(signal, noise_std=NOISE_STD, seed=42):
    rng = np.random.default_rng(seed)
    return signal + rng.normal(0, noise_std, size=len(signal))

def signal_to_bits(signal):
    return (np.clip(normalise(signal), 0, 1) > 0.5).astype(int)

def _spb(n_samples, n_bits):
    return max(1, n_samples // max(1, n_bits))

def snr_db(clean, noisy):
    sp  = np.mean(clean ** 2)
    np_ = np.mean((clean - noisy) ** 2)
    if np_ == 0: return float("inf")
    if sp  == 0: return float("-inf")
    return 10 * np.log10(sp / np_)

def ber(original_bits, recovered_bits):
    n = min(len(original_bits), len(recovered_bits))
    return float(np.sum(original_bits[:n] != recovered_bits[:n]) / n)

def _expand_bits(bits, n_samples):
    s   = _spb(n_samples, len(bits))
    out = np.repeat(bits.astype(float), s)[:n_samples]
    if len(out) < n_samples:
        out = np.pad(out, (0, n_samples - len(out)), constant_values=out[-1])
    return out

def channel(modulated, noise_std=NOISE_STD):
    return awgn(modulated, noise_std)


# =============================================================================
# SECTION 2 — AM
# =============================================================================

def am_modulate(baseband, fc=FC, fs=FS, mod_index=MOD_INDEX):
    x = normalise(baseband)
    t = time_axis(len(x), fs)
    return (1 + mod_index * x) * np.cos(2 * np.pi * fc * t)

def am_demodulate(received):
    env = np.abs(hilbert(received))
    env -= env.mean()
    return np.clip(normalise(env), 0, 1)


# =============================================================================
# SECTION 3 — FM
# =============================================================================

def fm_modulate(baseband, fc=FC, fs=FS, kf=KF):
    x     = normalise(baseband)
    t     = time_axis(len(x), fs)
    integ = np.cumsum(x) / fs
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integ)

def fm_demodulate(received, fc=FC, fs=FS):
    phase     = np.unwrap(np.angle(hilbert(received)))
    inst_freq = np.diff(phase) / (2 * np.pi / fs)
    inst_freq = np.append(inst_freq, inst_freq[-1]) - fc
    return np.clip(normalise(inst_freq), 0, 1)


# =============================================================================
# SECTION 4 — ASK
# =============================================================================

def ask_modulate(bits, n_samples, fc=FC, fs=FS):
    t   = time_axis(n_samples, fs)
    env = _expand_bits(bits, n_samples)
    return env * np.cos(2 * np.pi * fc * t)

def ask_demodulate(received, bits_orig):
    n   = len(received)
    nb  = len(bits_orig)
    s   = _spb(n, nb)
    env = np.abs(hilbert(received))
    recovered = np.array([
        1 if np.mean(env[i*s:(i+1)*s]) > 0.5 else 0
        for i in range(nb)
    ], dtype=int)
    return recovered, np.clip(normalise(env), 0, 1)


# =============================================================================
# SECTION 5 — FSK
# =============================================================================

def fsk_modulate(bits, n_samples, f0=F0_FSK, f1=F1_FSK, fs=FS):
    s   = _spb(n_samples, len(bits))
    t   = time_axis(n_samples, fs)
    out = np.zeros(n_samples)
    for i, b in enumerate(bits):
        st, en = i*s, min((i+1)*s, n_samples)
        out[st:en] = np.cos(2 * np.pi * (f1 if b else f0) * t[st:en])
    return out

def fsk_demodulate(received, bits_orig, f0=F0_FSK, f1=F1_FSK, fs=FS):
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
# SECTION 6 — PSK
# =============================================================================

def psk_modulate(bits, n_samples, fc=FC, fs=FS):
    t         = time_axis(n_samples, fs)
    phase_seq = _expand_bits(np.where(bits == 1, 0.0, np.pi), n_samples)
    return np.cos(2 * np.pi * fc * t + phase_seq)

def psk_demodulate(received, bits_orig, fc=FC, fs=FS):
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
# SECTION 7 — PER-SCHEME RUNNERS  (imported by generate_* and demodulation.py)
# =============================================================================

def run_am(baseband):
    mod = am_modulate(baseband)
    rx  = channel(mod)
    dem = am_demodulate(rx)
    return mod, rx, dem, snr_db(mod, rx)

def run_fm(baseband):
    mod = fm_modulate(baseband)
    rx  = channel(mod)
    dem = fm_demodulate(rx)
    return mod, rx, dem, snr_db(mod, rx)

def run_ask(baseband):
    bits     = signal_to_bits(baseband)
    mod      = ask_modulate(bits, len(baseband))
    rx       = channel(mod)
    rec, env = ask_demodulate(rx, bits)
    return mod, rx, rec, env, snr_db(mod, rx), ber(bits, rec), bits

def run_fsk(baseband):
    bits = signal_to_bits(baseband)
    mod  = fsk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = fsk_demodulate(rx, bits)
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits

def run_psk(baseband):
    bits = signal_to_bits(baseband)
    mod  = psk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = psk_demodulate(rx, bits)
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits


# =============================================================================
# SECTION 8 — STATION PROCESSING  (scheme × source × column)
# =============================================================================

def _load_inputs(station):
    """
    Load both input files for a station.
    Returns (df_filtered, df_features, filt_cols, feat_cols, filt_index, feat_index).
    filt_index = timestamp column, feat_index = segment_idx column.
    """
    fp = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    xp = os.path.join(PROCESSED_PATH, f"{station}_features.csv")

    df_filt = pd.read_csv(fp, parse_dates=["timestamp"]) if os.path.exists(fp) else None
    df_feat = pd.read_csv(xp) if os.path.exists(xp) else None

    filt_cols = [c for c in SIGNAL_COLS_FILTERED.get(station, [])
                 if df_filt is not None and c in df_filt.columns]
    feat_cols = [c for c in SIGNAL_COLS_FEATURES.get(station, [])
                 if df_feat is not None and c in df_feat.columns]

    return df_filt, df_feat, filt_cols, feat_cols


def process_station(station):
    """
    Run every (source × column × scheme) combination independently.
    source = filtered | features
    Returns list of metric dicts.
    """
    df_filt, df_feat, filt_cols, feat_cols = _load_inputs(station)
    results = []

    # Build list of (source_label, dataframe, columns, label_col)
    sources = []
    if df_filt is not None and filt_cols:
        sources.append(("filtered", df_filt, filt_cols, "timestamp"))
    if df_feat is not None and feat_cols:
        sources.append(("features", df_feat, feat_cols, "segment_idx"))

    for scheme in SCHEMES:
        print(f"  [{scheme}]")
        for src_label, df, cols, idx_col in sources:
            for col in cols:
                x = df[col].fillna(0).to_numpy(dtype=float)
                if len(x) < 10:
                    continue

                row = {"station": station, "source": src_label,
                       "signal_col": col, "scheme": scheme}

                if scheme == "AM":
                    mod, rx, dem, snr = run_am(x)
                    row.update({"SNR_dB": round(snr, 3), "BER": None})
                elif scheme == "FM":
                    mod, rx, dem, snr = run_fm(x)
                    row.update({"SNR_dB": round(snr, 3), "BER": None})
                elif scheme == "ASK":
                    mod, rx, rec, env, snr, ber_val, _ = run_ask(x)
                    row.update({"SNR_dB": round(snr, 3), "BER": round(ber_val, 5)})
                elif scheme == "FSK":
                    mod, rx, rec, snr, ber_val, _ = run_fsk(x)
                    row.update({"SNR_dB": round(snr, 3), "BER": round(ber_val, 5)})
                elif scheme == "PSK":
                    mod, rx, rec, snr, ber_val, _ = run_psk(x)
                    row.update({"SNR_dB": round(snr, 3), "BER": round(ber_val, 5)})

                results.append(row)
                ber_str = f"BER={row['BER']:.5f}" if row["BER"] is not None else "BER=N/A"
                print(f"    [{src_label:8s}] {col:35s} SNR={row['SNR_dB']:7.2f} dB  {ber_str}")

    return results


# =============================================================================
# SECTION 9 — SUMMARY PLOT
# =============================================================================

_BG   = "#0d1117"
_TEXT = "#e6edf3"

def plot_summary(all_results):
    if not all_results:
        return
    df       = pd.DataFrame(all_results)
    stations = sorted(df["station"].unique())
    x  = np.arange(len(stations))
    w  = 0.15
    colors = ["#3fb950","#f0883e","#bc8cff","#ff7b72","#ffa657"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=_BG)
    fig.suptitle("Modulation Performance — All Stations × Both Sources\nTELE 523 · Group 1",
                 fontsize=14, fontweight="bold", color=_TEXT)

    ax1 = axes[0]; ax1.set_facecolor("#161b22")
    for i, (s, c) in enumerate(zip(SCHEMES, colors)):
        vals = [df[(df.station==st) & (df.scheme==s)]["SNR_dB"].replace(
                [float("inf"), float("-inf")], np.nan).mean()
                for st in stations]
        bars = ax1.bar(x + i*w, vals, w, label=s, color=c, alpha=0.85, edgecolor="#ffffff11")
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax1.text(b.get_x()+b.get_width()/2, h+0.2, f"{h:.1f}",
                         ha="center", va="bottom", color=_TEXT, fontsize=5.5)
    ax1.set_xticks(x+2*w); ax1.set_xticklabels(stations, color=_TEXT, fontsize=8)
    ax1.set_ylabel("SNR (dB)", color="#8b949e")
    ax1.set_title("Mean SNR per Station & Scheme (filtered + features)", color=_TEXT, fontsize=10)
    ax1.legend(facecolor="#1c2128", labelcolor=_TEXT, fontsize=8)
    ax1.tick_params(colors="#8b949e")
    for sp in ax1.spines.values(): sp.set_color("#30363d")

    ax2 = axes[1]; ax2.set_facecolor("#161b22")
    dig = ["ASK","FSK","PSK"]; dc = ["#bc8cff","#ff7b72","#ffa657"]; w2=0.25
    for i,(s,c) in enumerate(zip(dig,dc)):
        vals = [df[(df.station==st)&(df.scheme==s)]["BER"].mean() for st in stations]
        bars = ax2.bar(x+i*w2, vals, w2, label=s, color=c, alpha=0.85, edgecolor="#ffffff11")
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax2.text(b.get_x()+b.get_width()/2, h+0.001, f"{h:.4f}",
                         ha="center", va="bottom", color=_TEXT, fontsize=5.5)
    ax2.set_xticks(x+w2); ax2.set_xticklabels(stations, color=_TEXT, fontsize=8)
    ax2.set_ylabel("BER", color="#8b949e")
    ax2.set_title("Mean BER per Station & Digital Scheme", color=_TEXT, fontsize=10)
    ax2.legend(facecolor="#1c2128", labelcolor=_TEXT, fontsize=8)
    ax2.tick_params(colors="#8b949e")
    for sp in ax2.spines.values(): sp.set_color("#30363d")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "modulation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"Summary plot saved -> {out}")


# =============================================================================
# SECTION 10 — MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TELE 523 - Modulation Lead (Student 3)")
    print("Inputs: *_filtered.csv (time-series) + *_features.csv (segments)")
    print("Each file × each column × each scheme — independently")
    print("=" * 70)
    print(f"  fc={FC} Hz | fs={FS} Hz | Nyquist={FS/2} Hz | AWGN sigma={NOISE_STD}\n")

    all_results = []
    for station in STATIONS:
        print(f"\n-- {station} " + "-"*50)
        all_results.extend(process_station(station))

    if not all_results:
        print("[ERROR] No results. Check data/processed/ for input files.")
        return

    df = pd.DataFrame(all_results)
    csv_out = os.path.join(OUT_DIR, "modulation_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nMetrics CSV -> {csv_out}")
    plot_summary(all_results)

    print("\n" + "="*70)
    print("SNR (dB) mean per station x scheme (filtered + features combined):")
    pivot = df.pivot_table(values="SNR_dB", index="station",
                           columns="scheme", aggfunc="mean").round(2)
    print(pivot.to_string())
    print("\nBER mean per station x scheme:")
    pivot_ber = df[df["BER"].notna()].pivot_table(
        values="BER", index="station", columns="scheme", aggfunc="mean").round(5)
    print(pivot_ber.to_string())
    print("\nModulation complete.")


if __name__ == "__main__":
    main()