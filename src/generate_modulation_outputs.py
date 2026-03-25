"""
generate_modulation_outputs.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Writes the handoff CSV files for Student 4 (Digital Telemetry Lead).

Imports all modulation functions from modulation.py — no math is duplicated
here. This script is purely responsible for:
  1. Running every signal column through every scheme independently
     (by calling the run_* functions from modulation.py)
  2. Saving the results in a format Student 4 can directly use

Output: 5 CSV files per station = 35 files total in results/modulation/output/
  {STATION}_AM.csv   — AM demodulated analog signal  [0,1]  -> quantize directly
  {STATION}_FM.csv   — FM demodulated analog signal  [0,1]  -> quantize directly
  {STATION}_ASK.csv  — ASK recovered bit sequence    {0,1}  -> line coding / PCM
  {STATION}_FSK.csv  — FSK recovered bit sequence    {0,1}  -> line coding / PCM
  {STATION}_PSK.csv  — PSK recovered bit sequence    {0,1}  -> line coding / PCM

Each file includes:
  - timestamp
  - {col}_baseband           normalised [0,1] source signal
  - {col}_*_demodulated  or  {col}_original_bits / {col}_*_recovered
  - current_state_binary     machine state label (ready=1) carried through
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Import all modulation functions from modulation.py ────────────────────────
# Both scripts live in src/. If running from the project root, the fallback
# sys.path insert handles discovery automatically.
try:
    from modulation import (
        SIGNAL_COLS,
        normalise, _expand_bits,
        run_am, run_fm, run_ask, run_fsk, run_psk,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modulation import (
        SIGNAL_COLS,
        normalise, _expand_bits,
        run_am, run_fm, run_ask, run_fsk, run_psk,
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR        = os.path.join(BASE_DIR, "results", "modulation", "output")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def generate_outputs():
    print("=" * 70)
    print("Generating Student 4 handoff CSVs")
    print("Every signal column passes through every scheme independently")
    print("=" * 70)

    total_files = 0

    for station in sorted(SIGNAL_COLS.keys()):
        fpath = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
        if not os.path.exists(fpath):
            print(f"[SKIP] {station} — filtered CSV not found")
            continue

        df       = pd.read_csv(fpath, parse_dates=["timestamp"])
        sig_cols = [c for c in SIGNAL_COLS[station] if c in df.columns]
        labels   = (df["current_state_binary"].fillna(0).astype(int)
                    if "current_state_binary" in df.columns
                    else pd.Series(np.zeros(len(df), dtype=int)))
        ts = df["timestamp"]

        print(f"\n-- {station} " + "-"*50)

        # One output dict per scheme — built up column by column
        am_out  = {"timestamp": ts, "current_state_binary": labels}
        fm_out  = {"timestamp": ts, "current_state_binary": labels}
        ask_out = {"timestamp": ts, "current_state_binary": labels}
        fsk_out = {"timestamp": ts, "current_state_binary": labels}
        psk_out = {"timestamp": ts, "current_state_binary": labels}

        for col in sig_cols:
            x = df[col].fillna(0).to_numpy(dtype=float)
            if len(x) < 30:
                continue
            n = len(x)

            # AM — independent run for this column
            _, _, dem_am, snr_am = run_am(x)
            am_out[f"{col}_baseband"]       = np.round(normalise(x), 6)
            am_out[f"{col}_am_demodulated"] = np.round(dem_am, 6)
            print(f"  AM  · {col:30s} SNR={snr_am:.1f} dB")

            # FM — independent run for this column
            _, _, dem_fm, snr_fm = run_fm(x)
            fm_out[f"{col}_baseband"]       = np.round(normalise(x), 6)
            fm_out[f"{col}_fm_demodulated"] = np.round(dem_fm, 6)
            print(f"  FM  · {col:30s} SNR={snr_fm:.1f} dB")

            # ASK — independent run for this column
            _, _, rec_ask, env_ask, snr_ask, ber_ask, orig_ask = run_ask(x)
            ask_out[f"{col}_baseband"]      = np.round(normalise(x), 6)
            ask_out[f"{col}_original_bits"] = _expand_bits(orig_ask, n).astype(int)
            ask_out[f"{col}_ask_recovered"] = _expand_bits(rec_ask,  n).astype(int)
            print(f"  ASK · {col:30s} SNR={snr_ask:.1f} dB | BER={ber_ask:.5f}")

            # FSK — independent run for this column
            _, _, rec_fsk, snr_fsk, ber_fsk, orig_fsk = run_fsk(x)
            fsk_out[f"{col}_baseband"]      = np.round(normalise(x), 6)
            fsk_out[f"{col}_original_bits"] = _expand_bits(orig_fsk, n).astype(int)
            fsk_out[f"{col}_fsk_recovered"] = _expand_bits(rec_fsk,  n).astype(int)
            print(f"  FSK · {col:30s} SNR={snr_fsk:.1f} dB | BER={ber_fsk:.5f}")

            # PSK — independent run for this column
            _, _, rec_psk, snr_psk, ber_psk, orig_psk = run_psk(x)
            psk_out[f"{col}_baseband"]      = np.round(normalise(x), 6)
            psk_out[f"{col}_original_bits"] = _expand_bits(orig_psk, n).astype(int)
            psk_out[f"{col}_psk_recovered"] = _expand_bits(rec_psk,  n).astype(int)
            print(f"  PSK · {col:30s} SNR={snr_psk:.1f} dB | BER={ber_psk:.5f}")

        # Save one CSV per scheme
        for scheme_name, data_dict in [("AM",  am_out),
                                        ("FM",  fm_out),
                                        ("ASK", ask_out),
                                        ("FSK", fsk_out),
                                        ("PSK", psk_out)]:
            df_out   = pd.DataFrame(data_dict)
            mid_cols = [c for c in df_out.columns
                        if c not in ("timestamp", "current_state_binary")]
            df_out   = df_out[["timestamp"] + mid_cols + ["current_state_binary"]]
            out_path = os.path.join(OUT_DIR, f"{station}_{scheme_name}.csv")
            df_out.to_csv(out_path, index=False)
            print(f"  Saved {station}_{scheme_name}.csv  "
                  f"({len(df_out)} rows x {len(df_out.columns)} cols)")
            total_files += 1

    print(f"\nDone — {total_files} handoff files written to:\n  {OUT_DIR}")


if __name__ == "__main__":
    generate_outputs()