"""
Phase 2 — Signal Conditioning & Feature Extraction
====================================================
Steps:
  9.  Sampling  — verify Nyquist & uniform 2-second resample
  10. Filtering  — FIR low-pass filter (replaces Butterworth IIR)
  11. Segmentation — sliding window (30-sample window, 50 % overlap)
  12. PSD        — Welch power-spectral-density per segment
  13. Features   — RMS, mean, variance, FFT peak, PSD peak, PSD mean energy

Input:  data/processed/<STATION>.csv          (Phase 1 output)
Output: data/processed/<STATION>_filtered.csv  (FIR-filtered, full signal)
        data/processed/<STATION>_features.csv  (one row per segment)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import firwin, lfilter, welch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH  = os.path.join(BASE_DIR, "data", "processed")
RESULTS_PATH    = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

# ── Sampling config ───────────────────────────────────────────────────────────
FS              = 0.5           # Hz  — one sample every 2 seconds
NYQUIST         = FS / 2        # 0.25 Hz
CUTOFF          = 0.1           # Hz  — low-pass cutoff frequency
FIR_NUMTAPS     = 51            # must be odd for Type-I linear phase FIR

# ── Segmentation config ───────────────────────────────────────────────────────
WINDOW_SIZE     = 30            # samples  (~60 s of data per segment)
OVERLAP         = 15            # samples  (50 % overlap)
STEP            = WINDOW_SIZE - OVERLAP   # = 15 samples between segment starts

# ── Welch PSD config ──────────────────────────────────────────────────────────
WELCH_NPERSEG   = min(WINDOW_SIZE, 16)    # sub-segment length for Welch

# ── Signal columns per station ────────────────────────────────────────────────
SIGNAL_COLS = {
    "EC_1"  : ["i3_photoresistor", "current_task_duration"],
    "HBW_1" : ["m1_speed", "m2_speed", "m3_speed", "m4_speed",
                "current_pos_x", "current_pos_y",
                "target_pos_x", "target_pos_y", "current_task_duration"],
    "MM_1"  : ["m1_speed", "m2_speed", "m3_speed",
                "o8_compressor", "current_task_duration"],
    "OV_1"  : ["m1_speed", "o8_compressor", "current_task_duration"],
    "SM_1"  : ["m1_speed", "o8_compressor", "current_task_duration"],
    "VGR_1" : ["m1_speed", "m2_speed", "m3_speed",
                "current_pos_x", "current_pos_y", "current_pos_z",
                "target_pos_x", "target_pos_y", "target_pos_z",
                "current_task_duration"],
    "WT_1"  : ["m2_speed", "o8_compressor", "current_task_duration"],
}


# ══════════════════════════════════════════════════════════════════════════════
# ── Helper: FIR low-pass filter ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def fir_lowpass_filter(signal: pd.Series,
                       cutoff: float,
                       fs: float,
                       numtaps: int) -> np.ndarray:
    """
    Apply a Type-I linear-phase FIR low-pass filter.

    Parameters
    ----------
    signal   : input signal (pd.Series)
    cutoff   : cut-off frequency in Hz
    fs       : sampling rate in Hz
    numtaps  : number of FIR taps (must be odd for Type-I)

    Returns
    -------
    Filtered signal as np.ndarray (same length as input).

    Why FIR over IIR?
    -----------------
    FIR filters have:
    • Guaranteed linear phase — no frequency-dependent delay distortion.
    • Inherent BIBO stability — no feedback path, so no pole instability.
    • Predictable transient behaviour — important for short industrial segments.
    IIR (Butterworth) filters are computationally cheaper but introduce
    non-linear phase and can become unstable with aggressive cutoffs.
    """
    nyquist     = fs / 2.0
    normal_cut  = cutoff / nyquist          # normalise to [0, 1]
    # firwin designs a windowed-sinc FIR; default window = Hamming
    fir_coefs   = firwin(numtaps, normal_cut, window="hamming")
    return lfilter(fir_coefs, 1.0, signal.values)


# ══════════════════════════════════════════════════════════════════════════════
# ── Helper: Welch PSD features for one segment ───────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def welch_psd_features(segment: np.ndarray,
                       col_name: str,
                       fs: float,
                       nperseg: int) -> dict:
    """
    Compute Welch power-spectral-density and extract scalar features.

    Returns
    -------
    dict with:
      {col}_psd_peak      — maximum PSD value (dominant frequency power)
      {col}_psd_mean_energy — mean PSD across all bins (average power)
      {col}_psd_peak_freq — frequency (Hz) at which PSD is maximum
    """
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
    peak_idx   = np.argmax(psd)
    return {
        f"{col_name}_psd_peak"        : float(np.max(psd)),
        f"{col_name}_psd_mean_energy" : float(np.mean(psd)),
        f"{col_name}_psd_peak_freq"   : float(freqs[peak_idx]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Helper: time-domain + FFT features for one segment ───────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def time_domain_features(segment: np.ndarray, col_name: str) -> dict:
    """
    Compute classic time-domain and FFT-peak features for one segment.

    Returns
    -------
    dict with: rms, mean, variance, fft_peak
    """
    fft_vals = np.abs(np.fft.rfft(segment))
    return {
        f"{col_name}_rms"      : float(np.sqrt(np.mean(segment ** 2))),
        f"{col_name}_mean"     : float(np.mean(segment)),
        f"{col_name}_variance" : float(np.var(segment)),
        f"{col_name}_fft_peak" : float(np.max(fft_vals)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Helper: sliding-window segmentation ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def sliding_windows(arr: np.ndarray,
                    window_size: int,
                    step: int):
    """
    Generator yielding (start_idx, segment_array) tuples from arr.

    Parameters
    ----------
    arr         : 1-D numpy array (single column, after FIR filtering)
    window_size : number of samples per window
    step        : number of samples to advance between windows

    Yields
    ------
    (start_idx, segment)  — segment is a view of arr[start:start+window_size]
    """
    n = len(arr)
    start = 0
    while start + window_size <= n:
        yield start, arr[start : start + window_size]
        start += step


# ══════════════════════════════════════════════════════════════════════════════
# ── Main processing loop ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
for station in sorted(SIGNAL_COLS.keys()):

    csv_path = os.path.join(PROCESSED_PATH, f"{station}.csv")
    sdf      = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"\n{'─'*60}")
    print(f"  Station : {station}   ({len(sdf)} rows)")
    print(f"{'─'*60}")

    sig_cols = [c for c in SIGNAL_COLS[station] if c in sdf.columns]

    # ── Step 9 · Resample to uniform 2-second grid ────────────────────────────
    sdf = sdf.set_index("timestamp").sort_index()
    sdf = sdf.resample("2s").mean(numeric_only=True).ffill()
    print(f"  Step 9  — resampled to 2 s grid | "
          f"fs={FS} Hz | Nyquist={NYQUIST} Hz | cutoff={CUTOFF} Hz ✓")

    # ── Step 10 · FIR low-pass filter ─────────────────────────────────────────
    # FIR filter requires len(signal) > numtaps; guard against tiny stations.
    effective_taps = min(FIR_NUMTAPS, len(sdf) // 2 * 2 - 1)
    if effective_taps < 3:
        effective_taps = 3

    for col in sig_cols:
        if sdf[col].nunique() > 1:          # skip constant / all-NaN columns
            sdf[col] = fir_lowpass_filter(
                sdf[col], CUTOFF, FS, numtaps=effective_taps
            )
    print(f"  Step 10 — FIR low-pass filter applied "
          f"(numtaps={effective_taps}, cutoff={CUTOFF} Hz)")
    print(f"            Columns filtered: {sig_cols}")

    # Save FIR-filtered full signal
    sdf_out = sdf.reset_index()
    filtered_path = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    sdf_out.to_csv(filtered_path, index=False)
    print(f"  Saved FIR-filtered signal  → {station}_filtered.csv")

    # ── Steps 11-13 · Segmentation + PSD + Feature extraction ────────────────
    # One feature row is produced per segment per column.
    # All column features for the same segment window are merged into one row.
    all_segment_rows = []

    # We use the filtered signal arrays; target label derived from majority vote
    # of current_state_binary within the window.
    label_arr = sdf["current_state_binary"].values if "current_state_binary" in sdf.columns else None

    # Build a reference array of timestamps for segment metadata
    timestamps = sdf.index.values   # numpy datetime64 array

    # Count segments (based on first sig col, or full length if none)
    ref_arr = sdf[sig_cols[0]].values if sig_cols else np.zeros(len(sdf))
    n_segments = max(0, (len(ref_arr) - WINDOW_SIZE) // STEP + 1)
    print(f"  Step 11 — Segmentation | window={WINDOW_SIZE} samples "
          f"({WINDOW_SIZE*2} s) | overlap={OVERLAP} samples | "
          f"→ {n_segments} segments")

    for start_idx, _ in sliding_windows(ref_arr, WINDOW_SIZE, STEP):
        end_idx      = start_idx + WINDOW_SIZE
        row_features = {}

        # Segment metadata
        row_features["segment_start"] = str(timestamps[start_idx])
        row_features["segment_end"]   = str(timestamps[end_idx - 1])
        row_features["segment_idx"]   = start_idx // STEP

        # Majority-vote label for the segment
        if label_arr is not None:
            seg_labels = label_arr[start_idx:end_idx]
            row_features["current_state_binary"] = int(
                np.round(np.nanmean(seg_labels))
            )

        # Per-column features
        for col in sig_cols:
            segment = sdf[col].values[start_idx:end_idx]

            # Step 12 — Welch PSD features
            psd_feats = welch_psd_features(
                segment, col, fs=FS, nperseg=WELCH_NPERSEG
            )
            row_features.update(psd_feats)

            # Step 13 — Time-domain + FFT features
            td_feats = time_domain_features(segment, col)
            row_features.update(td_feats)

        all_segment_rows.append(row_features)

    if not all_segment_rows:
        print(f"  ⚠  WARNING: No complete segments for {station} "
              f"(signal too short for window={WINDOW_SIZE}). "
              f"Skipping feature export.")
        continue

    features_df   = pd.DataFrame(all_segment_rows)
    features_path = os.path.join(PROCESSED_PATH, f"{station}_features.csv")
    features_df.to_csv(features_path, index=False)

    feature_cols = [c for c in features_df.columns
                    if c not in ("segment_start", "segment_end",
                                 "segment_idx", "current_state_binary")]
    print(f"  Step 12 — Welch PSD computed per segment "
          f"(nperseg={WELCH_NPERSEG})")
    print(f"  Step 13 — Features extracted per segment: "
          f"{len(feature_cols)} feature columns")
    print(f"  Saved features ({len(features_df)} rows) "
          f"→ {station}_features.csv")

print(f"\n{'═'*60}")
print("Phase 2 complete — FIR-filtered signals + segmented features")
print("Handing over to Modulation Lead.")
print(f"{'═'*60}")

# ── Plot: all filtered signals per station → results/ ─────────────────────────
print("\nGenerating filtered-signal charts…")
TAB20 = cm.get_cmap("tab20")

for station, sig_cols in sorted(SIGNAL_COLS.items()):
    filtered_csv = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    if not os.path.exists(filtered_csv):
        print(f"  ⚠  {station}_filtered.csv not found, skipping plot.")
        continue

    sdf = pd.read_csv(filtered_csv)
    cols = [c for c in sig_cols if c in sdf.columns]
    if not cols:
        continue

    n = len(cols)
    row_h = max(1.4, 90 / n)
    fig, axes = plt.subplots(n, 1, figsize=(18, row_h * n), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Station {station} — {n} signals  [SIGNAL PROCESSING · FIR filtered, 2 s grid]",
        fontsize=13, fontweight="bold", color="white", y=1.002,
    )

    x = np.arange(len(sdf))
    for i, (ax, col) in enumerate(zip(axes, cols)):
        color = TAB20(i / max(n, 1))
        vals = pd.to_numeric(sdf[col], errors="coerce").to_numpy()
        ax.set_facecolor("#161b22")
        ax.plot(x, vals, linewidth=0.9, color=color)
        ax.set_ylabel(col, fontsize=7, rotation=0, labelpad=2,
                      ha="right", va="center", color="#cccccc")
        ax.tick_params(axis="both", labelsize=6, colors="#aaaaaa")
        ax.spines[:].set_color("#333333")
        ax.grid(axis="x", linewidth=0.3, alpha=0.4, color="#444444")

        if "current_state_binary" in sdf.columns:
            ready_mask = sdf["current_state_binary"].astype(float) == 1
            ax.fill_between(x, 0, 1, where=ready_mask,
                            color="#2ebd7a", alpha=0.07,
                            transform=ax.get_xaxis_transform())

    axes[-1].set_xlabel("Sample index", fontsize=8, color="#aaaaaa")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_PATH, f"{station}_filtered_signals.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → results/{station}_filtered_signals.png")

print("Signal processing charts saved to results/")

# ── Combined overview: all stations, all signals overlaid ─────────────────────
print("\nGenerating combined overview chart…")
stations_sorted = sorted(SIGNAL_COLS.keys())
n_stations = len(stations_sorted)

fig_all, axes_all = plt.subplots(n_stations, 1, figsize=(20, 5 * n_stations), sharex=False)
fig_all.patch.set_facecolor("#0d1117")
fig_all.suptitle(
    "All Stations — All Filtered Signals  [SIGNAL PROCESSING · FIR filtered, 2 s grid]",
    fontsize=15, fontweight="bold", color="white", y=1.002,
)

for ax, station in zip(axes_all, stations_sorted):
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#333333")
    ax.tick_params(colors="#aaaaaa", labelsize=7)

    filtered_csv = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    if not os.path.exists(filtered_csv):
        ax.text(0.5, 0.5, f"{station} — filtered CSV not found",
                ha="center", va="center", color="#ff6666",
                transform=ax.transAxes, fontsize=10)
        ax.set_title(station, color="white", fontsize=11)
        continue

    sdf = pd.read_csv(filtered_csv)
    cols = [c for c in SIGNAL_COLS[station] if c in sdf.columns]
    x = np.arange(len(sdf))
    colors_cycle = TAB20(np.linspace(0, 1, max(len(cols), 1)))

    for col, color in zip(cols, colors_cycle):
        vals = pd.to_numeric(sdf[col], errors="coerce").to_numpy()
        ax.plot(x, vals, linewidth=0.8, alpha=0.8, label=col, color=color)

    ax.set_title(f"Station: {station}  ({len(cols)} signals)", color="white",
                 fontsize=11, pad=5)
    ax.set_xlabel("Sample index", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Filtered value", color="#aaaaaa", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=3,
              facecolor="#1e2530", labelcolor="white", framealpha=0.6)

plt.tight_layout()
combined_path = os.path.join(RESULTS_PATH, "all_filtered_signals.png")
fig_all.savefig(combined_path, dpi=150, bbox_inches="tight",
                facecolor=fig_all.get_facecolor())
plt.close(fig_all)
print("  Saved → results/all_filtered_signals.png")