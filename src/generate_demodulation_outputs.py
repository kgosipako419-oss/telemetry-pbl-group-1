"""
generate_demodulation_outputs.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Produces 70 demodulation handoff CSV files for Student 4.
  7 stations × 2 sources (filtered + features) × 5 schemes = 70 files

Output naming:
  {STATION}_filtered_{SCHEME}_demod.csv
  {STATION}_features_{SCHEME}_demod.csv

Column layout — filtered (analog AM/FM):
  timestamp | {col}_baseband | {col}_{scheme}_demodulated | ... | current_state_binary

Column layout — filtered (digital ASK/FSK/PSK):
  timestamp | {col}_baseband | {col}_original_bits | {col}_{scheme}_recovered | ... | current_state_binary

Column layout — features (analog AM/FM):
  segment_start | segment_end | segment_idx | {col}_baseband | {col}_{scheme}_demodulated | ... | current_state_binary

Column layout — features (digital ASK/FSK/PSK):
  segment_start | segment_end | segment_idx | {col}_baseband | {col}_original_bits | {col}_{scheme}_recovered | ... | current_state_binary

Student 4 usage:
  AM/FM      -> uniform quantization + PCM encoding on *_demodulated columns
  ASK/FSK/PSK -> NRZ-L / Manchester line coding on *_recovered columns
"""

import os, sys
import numpy as np
import pandas as pd

try:
    from demodulation import (
        STATIONS, SIGNAL_COLS_FILTERED, SIGNAL_COLS_FEATURES, PROCESSED_PATH,
        normalise, _expand_bits,
        pipeline_am, pipeline_fm, pipeline_ask, pipeline_fsk, pipeline_psk,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from demodulation import (
        STATIONS, SIGNAL_COLS_FILTERED, SIGNAL_COLS_FEATURES, PROCESSED_PATH,
        normalise, _expand_bits,
        pipeline_am, pipeline_fm, pipeline_ask, pipeline_fsk, pipeline_psk,
    )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "results", "demodulation", "output")
os.makedirs(OUT_DIR, exist_ok=True)

SCHEME_PIPELINES = {
    "AM" : pipeline_am,
    "FM" : pipeline_fm,
    "ASK": pipeline_ask,
    "FSK": pipeline_fsk,
    "PSK": pipeline_psk,
}


def _build_demod_dict(df, cols, index_col, scheme):
    """Run every column through one demodulation pipeline and return output dict.

    For filtered files  : index_col = 'timestamp'
    For features files  : index_col = 'segment_idx',
                          also carries segment_start + segment_end
    """
    out = {index_col: df[index_col]}

    # ── carry segment_start / segment_end when present (features files) ───────
    if "segment_start" in df.columns:
        out["segment_start"] = df["segment_start"]
    if "segment_end" in df.columns:
        out["segment_end"] = df["segment_end"]

    out["current_state_binary"] = df["current_state_binary"].fillna(0).astype(int) \
        if "current_state_binary" in df.columns \
        else pd.Series(np.zeros(len(df), dtype=int))

    for col in cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 10:
            continue
        n = len(x)
        out[f"{col}_baseband"] = np.round(normalise(x), 6)

        if scheme == "AM":
            _, _, dem, snr = pipeline_am(x)
            out[f"{col}_am_demodulated"] = np.round(dem, 6)
            print(f"    AM  · {col:40s} SNR={snr:6.2f} dB")

        elif scheme == "FM":
            _, _, dem, snr = pipeline_fm(x)
            out[f"{col}_fm_demodulated"] = np.round(dem, 6)
            print(f"    FM  · {col:40s} SNR={snr:6.2f} dB")

        elif scheme == "ASK":
            _, _, rec, _, snr, ber_val, orig = pipeline_ask(x)
            out[f"{col}_original_bits"] = _expand_bits(orig, n).astype(int)
            out[f"{col}_ask_recovered"] = _expand_bits(rec,  n).astype(int)
            print(f"    ASK · {col:40s} SNR={snr:6.2f} dB  BER={ber_val:.5f}")

        elif scheme == "FSK":
            _, _, rec, snr, ber_val, orig = pipeline_fsk(x)
            out[f"{col}_original_bits"] = _expand_bits(orig, n).astype(int)
            out[f"{col}_fsk_recovered"] = _expand_bits(rec,  n).astype(int)
            print(f"    FSK · {col:40s} SNR={snr:6.2f} dB  BER={ber_val:.5f}")

        elif scheme == "PSK":
            _, _, rec, snr, ber_val, orig = pipeline_psk(x)
            out[f"{col}_original_bits"] = _expand_bits(orig, n).astype(int)
            out[f"{col}_psk_recovered"] = _expand_bits(rec,  n).astype(int)
            print(f"    PSK · {col:40s} SNR={snr:6.2f} dB  BER={ber_val:.5f}")

    return out


def _save(data_dict, index_col, path):
    df_out = pd.DataFrame(data_dict)

    # Fixed leading cols: index first, then segment times if present, then label
    leading = [index_col]
    for extra in ("segment_start", "segment_end"):
        if extra in df_out.columns:
            leading.append(extra)
    trailing = ["current_state_binary"]

    mid_cols = [c for c in df_out.columns if c not in leading + trailing]
    df_out   = df_out[leading + mid_cols + trailing]
    df_out.to_csv(path, index=False)
    return df_out.shape


def generate_demodulation_outputs():
    print("=" * 70)
    print("TELE 523 · Student 3 — Demodulation Output Generator")
    print("14 input files × 5 schemes = 70 demodulated handoff files")
    print("=" * 70)
    total = 0

    for station in STATIONS:
        print(f"\n-- {station} " + "-"*50)

        # ── filtered source ───────────────────────────────────────────────────
        fp = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
        if os.path.exists(fp):
            df_filt   = pd.read_csv(fp, parse_dates=["timestamp"])
            filt_cols = [c for c in SIGNAL_COLS_FILTERED.get(station, [])
                         if c in df_filt.columns]
            for scheme in ["AM", "FM", "ASK", "FSK", "PSK"]:
                print(f"  [filtered][{scheme}]")
                out  = _build_demod_dict(df_filt, filt_cols, "timestamp", scheme)
                path = os.path.join(OUT_DIR, f"{station}_filtered_{scheme}_demod.csv")
                rows, cols = _save(out, "timestamp", path)
                print(f"  --> Saved {os.path.basename(path)}  ({rows} rows × {cols} cols)")
                total += 1

        # ── features source ───────────────────────────────────────────────────
        xp = os.path.join(PROCESSED_PATH, f"{station}_features.csv")
        if os.path.exists(xp):
            df_feat = pd.read_csv(xp)
            if "segment_idx" not in df_feat.columns:
                df_feat["segment_idx"] = range(len(df_feat))
            feat_cols = [c for c in SIGNAL_COLS_FEATURES.get(station, [])
                         if c in df_feat.columns]
            for scheme in ["AM", "FM", "ASK", "FSK", "PSK"]:
                print(f"  [features][{scheme}]")
                out  = _build_demod_dict(df_feat, feat_cols, "segment_idx", scheme)
                path = os.path.join(OUT_DIR, f"{station}_features_{scheme}_demod.csv")
                rows, cols = _save(out, "segment_idx", path)
                print(f"  --> Saved {os.path.basename(path)}  ({rows} rows × {cols} cols)")
                total += 1

    print(f"\n{'='*70}")
    print(f"DONE — {total} demodulation output files written to:")
    print(f"  {OUT_DIR}")
    print(f"\nStudent 4 handoff:")
    print("  *_filtered_AM_demod.csv  / *_features_AM_demod.csv   -> quantize AM analog [0,1]")
    print("  *_filtered_FM_demod.csv  / *_features_FM_demod.csv   -> quantize FM analog [0,1]")
    print("  *_filtered_ASK_demod.csv / *_features_ASK_demod.csv  -> NRZ/Manchester line coding")
    print("  *_filtered_FSK_demod.csv / *_features_FSK_demod.csv  -> NRZ/Manchester line coding")
    print("  *_filtered_PSK_demod.csv / *_features_PSK_demod.csv  -> NRZ/Manchester line coding")


if __name__ == "__main__":
    generate_demodulation_outputs()