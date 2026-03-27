# 📡 Telemetry PBL – Group 1
### TELE 523 · Industrial Machine Condition Monitoring
**Branch: `PREPROCESSING`** — Signal Processing Lead

---

## 🗂️ Overview

This branch covers **Phase 1 (Data Preprocessing)** and **Phase 2 (Signal Processing)** of the industrial machine condition monitoring pipeline. Raw telemetry logs from 7 factory stations are cleaned, filtered, segmented, and transformed into feature vectors ready for modulation and classification.

---

## 🏭 Dataset

| Property | Detail |
|---|---|
| **Source file** | `data/raw/7612698/low-level_log_20230206-140808.txt` |
| **Format** | JSON lines (one record per line) |
| **Date recorded** | 2023-02-06, 14:08 – 15:11 |
| **Stations** | EC_1, HBW_1, MM_1, OV_1, SM_1, VGR_1, WT_1 |
| **Total records** | ~13,190 |

### Station Descriptions

| Station | Full Name | Role |
|---|---|---|
| `EC_1` | Environment Controller | Monitors ambient conditions (light, temperature) |
| `HBW_1` | High-Bay Warehouse | Stores and retrieves workpieces |
| `MM_1` | Milling Machine | Machines workpieces |
| `OV_1` | Oven | Heat-treats workpieces |
| `SM_1` | Sorting Machine | Sorts finished workpieces |
| `VGR_1` | Vacuum Gripper Robot | Transfers workpieces between stations |
| `WT_1` | Work Transfer | Moves workpieces along the line |

---

## 📁 Project Structure

```
telemetry-pbl-group-1/
├── data/
│   ├── raw/                          ← original log file (not modified)
│   └── processed/
│       ├── EC_1.csv                  ← Phase 1 output (clean, normalised)
│       ├── EC_1_filtered.csv         ← Phase 2 output (FIR-filtered signal)
│       ├── EC_1_features.csv         ← Phase 2 output (segmented features)
│       └── ... (same pattern for all 7 stations)
├── results/
│   ├── PREPROCESSING_pipeline.png
│   ├── PREPROCESSING_results_summary.png
│   └── PREPROCESSING_interconnection_web.png
├── src/
│   ├── preprocessing.py              ← Phase 1 script
│   ├── signal_processing.py          ← Phase 2 script
│   └── generate_images.py            ← pipeline visualisation script
├── requirements.txt
└── README.md
```

---

## ⚙️ Phase 1 — Data Preprocessing (`src/preprocessing.py`)

Transforms raw JSON telemetry into clean, normalised, per-station CSVs.

| Step | Name | What it does |
|---|---|---|
| 1 | Load & Parse | Reads JSON lines into a Pandas DataFrame (~13,190 records) |
| 2 | Separate by station | Splits into 7 individual DataFrames keyed by `station` field |
| 3 | Parse timestamps | Converts `timestamp` to `datetime64`, sorts chronologically |
| 4 | Handle missing values | `ffill`/`bfill` on numeric cols; `fillna("")` on string cols |
| 5 | Encode categoricals | 26 boolean cols → `Int8` (0/1); 3 string cols → integer codes |
| 6 | Normalise signals | Min-max scaling to `[0, 1]` on all non-categorical numeric cols |
| 7 | Label target variable | `current_state == "ready"` → `current_state_binary = 1`, else `0` |
| 8 | Save CSVs | Writes 7 × 51-column CSVs to `data/processed/` |

### Phase 1 Output Summary

| Station | Rows | Columns |
|---|---|---|
| EC_1 | 1,886 | 51 |
| HBW_1 | 1,881 | 51 |
| MM_1 | 1,886 | 51 |
| OV_1 | 1,885 | 51 |
| SM_1 | 1,885 | 51 |
| VGR_1 | 1,882 | 51 |
| WT_1 | 1,885 | 51 |

---

## 📶 Phase 2 — Signal Processing (`src/signal_processing.py`)

Conditions the cleaned signals using FIR filtering, sliding-window segmentation, Welch PSD analysis, and multi-feature extraction.

### Configuration

| Parameter | Value | Meaning |
|---|---|---|
| `FS` | 0.5 Hz | Sampling rate (1 sample per 2 seconds) |
| `NYQUIST` | 0.25 Hz | Nyquist frequency |
| `CUTOFF` | 0.1 Hz | FIR low-pass cutoff frequency |
| `FIR_NUMTAPS` | 51 | Number of FIR filter taps (odd → Type-I linear phase) |
| `WINDOW_SIZE` | 30 samples | Segment length (~60 seconds of data) |
| `OVERLAP` | 15 samples | Overlap between windows (50%) |
| `STEP` | 15 samples | Stride between segment starts |
| `WELCH_NPERSEG` | 16 | Sub-segment length for Welch PSD estimation |

### Processing Steps

| Step | Name | What it does |
|---|---|---|
| 9 | Sampling | Resamples to uniform 2-second grid via `resample("2s").mean().ffill()` |
| 10 | FIR Low-Pass Filter | Applies `scipy.signal.firwin` (Hamming window) + `lfilter`; removes noise above 0.1 Hz with guaranteed linear phase |
| 11 | Segmentation | Sliding window over the filtered signal; each window = one training sample; label assigned by majority vote within window |
| 12 | Welch PSD | `scipy.signal.welch` per segment per signal column; reduces spectral variance by averaging overlapping sub-windows |
| 13 | Feature Extraction | Extracts 7 features per signal column per segment (see below) |

### Why FIR over IIR (Butterworth)?

| Property | FIR (used) | IIR Butterworth (old) |
|---|---|---|
| Phase response | Linear — no frequency-dependent delay | Non-linear — introduces distortion |
| Stability | Unconditionally stable (no feedback) | Can become unstable with aggressive cutoffs |
| Transient behaviour | Predictable on short segments | Ringing artefacts possible |
| Computational cost | Slightly higher | Lower |

### Features Extracted per Segment

For each signal column in each segment, 7 features are computed:

| Feature | Description |
|---|---|
| `_rms` | Root Mean Square — overall signal energy |
| `_mean` | Arithmetic mean — DC offset / average level |
| `_variance` | Statistical variance — signal spread |
| `_fft_peak` | Peak magnitude of the FFT spectrum |
| `_psd_peak` | Maximum value of the Welch PSD |
| `_psd_mean_energy` | Mean PSD across all frequency bins |
| `_psd_peak_freq` | Frequency (Hz) at which PSD is maximum |

### Phase 2 Output Summary

| Station | Signal cols | Segments | Feature cols per segment |
|---|---|---|---|
| EC_1 | 2 | ~125 | 14 |
| HBW_1 | 9 | ~124 | 63 |
| MM_1 | 5 | ~125 | 35 |
| OV_1 | 3 | ~125 | 21 |
| SM_1 | 3 | ~125 | 21 |
| VGR_1 | 10 | ~124 | 70 |
| WT_1 | 3 | ~125 | 21 |

Each `*_features.csv` has one row per segment, with columns:
`segment_start`, `segment_end`, `segment_idx`, `current_state_binary`, + all feature columns.

---

## 🖼️ Results (`src/generate_images.py`)

Generates three diagnostic figures saved to `results/`:

| File | Description |
|---|---|
| `PREPROCESSING_pipeline.png` | Full step-by-step pipeline diagram (Phase 1 + Phase 2) |
| `PREPROCESSING_results_summary.png` | Bar charts (records, segments, features) + step tables |
| `PREPROCESSING_interconnection_web.png` | Station data-flow web showing workpiece movement |

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Phase 1 — Preprocessing
```bash
python src/preprocessing.py
```

### Run Phase 2 — Signal Processing
```bash
python src/signal_processing.py
```

### Generate pipeline images
```bash
python src/generate_images.py
```

---

## 📦 Dependencies

```
pandas
numpy
scipy
matplotlib
```

---

## 👥 Authors

**TELE 523 · Group 1**
Signal Processing Lead — Phase 1 & Phase 2 pipeline

