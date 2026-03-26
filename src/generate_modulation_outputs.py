"""
generate_modulation_outputs.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Saves modulated signal CSVs for both input sources.

Inputs:
  data/processed/{STATION}_filtered.csv   — time-series baseband (1893 rows)
  data/processed/{STATION}_features.csv   — segment feature vectors (125 rows)

Outputs (70 files = 7 stations × 2 sources × 5 schemes):
  results/modulation/output/{STATION}_filtered_{SCHEME}.csv
  results/modulation/output/{STATION}_features_{SCHEME}.csv

All modulation math is imported from modulation.py.
"""

import os, sys
import numpy as np
import pandas as pd

try:
    from modulation import (
        STATIONS, SIGNAL_COLS_FILTERED, SIGNAL_COLS_FEATURES,
        PROCESSED_PATH, normalise, _expand_bits,
        run_am, run_fm, run_ask, run_fsk, run_psk,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modulation import (
        STATIONS, SIGNAL_COLS_FILTERED, SIGNAL_COLS_FEATURES,
        PROCESSED_PATH, normalise, _expand_bits,
        run_am, run_fm, run_ask, run_fsk, run_psk,
    )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "results", "modulation", "output")
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_META = ["segment_start", "segment_end", "segment_idx", "current_state_binary"]


def _build_scheme_dict(df, cols, index_col, scheme):
    """Run every column through one scheme and return output dict."""
    out = {index_col: df[index_col], "current_state_binary": df.get("current_state_binary", 0)}
    for col in cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 10:
            continue
        n = len(x)
        out[f"{col}_baseband"] = np.round(normalise(x), 6)

        if scheme == "AM":
            _, _, dem, _ = run_am(x)
            out[f"{col}_am_demodulated"] = np.round(dem, 6)
        elif scheme == "FM":
            _, _, dem, _ = run_fm(x)
            out[f"{col}_fm_demodulated"] = np.round(dem, 6)
        elif scheme == "ASK":
            _, _, rec, _, _, _, orig = run_ask(x)
            out[f"{col}_original_bits"]  = _expand_bits(orig, n).astype(int)
            out[f"{col}_ask_recovered"]  = _expand_bits(rec,  n).astype(int)
        elif scheme == "FSK":
            _, _, rec, _, _, orig = run_fsk(x)
            out[f"{col}_original_bits"]  = _expand_bits(orig, n).astype(int)
            out[f"{col}_fsk_recovered"]  = _expand_bits(rec,  n).astype(int)
        elif scheme == "PSK":
            _, _, rec, _, _, orig = run_psk(x)
            out[f"{col}_original_bits"]  = _expand_bits(orig, n).astype(int)
            out[f"{col}_psk_recovered"]  = _expand_bits(rec,  n).astype(int)
    return out


def _save(data_dict, index_col, path):
    df_out   = pd.DataFrame(data_dict)
    mid_cols = [c for c in df_out.columns
                if c not in (index_col, "current_state_binary")]
    df_out   = df_out[[index_col] + mid_cols + ["current_state_binary"]]
    df_out.to_csv(path, index=False)
    return df_out.shape


def generate_outputs():
    print("=" * 70)
    print("Generating modulation output CSVs")
    print("14 input files × 5 schemes = 70 output files")
    print("=" * 70)
    total = 0

    for station in STATIONS:
        print(f"\n-- {station} " + "-"*50)

        # --- filtered source ---
        fp = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
        if os.path.exists(fp):
            df_filt  = pd.read_csv(fp, parse_dates=["timestamp"])
            filt_cols = [c for c in SIGNAL_COLS_FILTERED.get(station, [])
                         if c in df_filt.columns]
            for scheme in ["AM","FM","ASK","FSK","PSK"]:
                out = _build_scheme_dict(df_filt, filt_cols, "timestamp", scheme)
                path = os.path.join(OUT_DIR, f"{station}_filtered_{scheme}.csv")
                rows, cols = _save(out, "timestamp", path)
                print(f"  [filtered][{scheme}]  {rows} rows × {cols} cols  -> {os.path.basename(path)}")
                total += 1

        # --- features source ---
        xp = os.path.join(PROCESSED_PATH, f"{station}_features.csv")
        if os.path.exists(xp):
            df_feat  = pd.read_csv(xp)
            if "segment_idx" not in df_feat.columns:
                df_feat["segment_idx"] = range(len(df_feat))
            feat_cols = [c for c in SIGNAL_COLS_FEATURES.get(station, [])
                         if c in df_feat.columns]
            for scheme in ["AM","FM","ASK","FSK","PSK"]:
                out = _build_scheme_dict(df_feat, feat_cols, "segment_idx", scheme)
                path = os.path.join(OUT_DIR, f"{station}_features_{scheme}.csv")
                rows, cols = _save(out, "segment_idx", path)
                print(f"  [features][{scheme}]  {rows} rows × {cols} cols  -> {os.path.basename(path)}")
                total += 1

    print(f"\nDone — {total} modulation output files in:\n  {OUT_DIR}")


if __name__ == "__main__":
    generate_outputs()