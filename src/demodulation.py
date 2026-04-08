"""
demodulation.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Standalone demodulation module. Imports all modulation math from modulation.py.

Exposes pipeline_* functions used by generate_demodulation_outputs.py.
Also computes and saves a full metrics summary when run directly.
"""

import os, sys
import numpy as np
import pandas as pd

try:
    from modulation import (
        STATIONS, SIGNAL_COLS_FILTERED, SIGNAL_COLS_FEATURES,
        PROCESSED_PATH, normalise, _expand_bits, _spb,
        snr_db, ber, signal_to_bits,
        am_modulate, am_demodulate,
        fm_modulate, fm_demodulate,
        ask_modulate, ask_demodulate,
        fsk_modulate, fsk_demodulate,
        psk_modulate, psk_demodulate,
        channel,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modulation import (
        STATIONS, SIGNAL_COLS_FILTERED, SIGNAL_COLS_FEATURES,
        PROCESSED_PATH, normalise, _expand_bits, _spb,
        snr_db, ber, signal_to_bits,
        am_modulate, am_demodulate,
        fm_modulate, fm_demodulate,
        ask_modulate, ask_demodulate,
        fsk_modulate, fsk_demodulate,
        psk_modulate, psk_demodulate,
        channel,
    )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "results", "demodulation")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# FULL PIPELINE FUNCTIONS  (modulate → channel → demodulate)
# =============================================================================

def pipeline_am(baseband):
    mod = am_modulate(baseband)
    rx  = channel(mod)
    dem = am_demodulate(rx)
    return mod, rx, dem, snr_db(mod, rx)

def pipeline_fm(baseband):
    mod = fm_modulate(baseband)
    rx  = channel(mod)
    dem = fm_demodulate(rx)
    return mod, rx, dem, snr_db(mod, rx)

def pipeline_ask(baseband):
    bits     = signal_to_bits(baseband)
    mod      = ask_modulate(bits, len(baseband))
    rx       = channel(mod)
    rec, env = ask_demodulate(rx, bits)
    return mod, rx, rec, env, snr_db(mod, rx), ber(bits, rec), bits

def pipeline_fsk(baseband):
    bits = signal_to_bits(baseband)
    mod  = fsk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = fsk_demodulate(rx, bits)
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits

def pipeline_psk(baseband):
    bits = signal_to_bits(baseband)
    mod  = psk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = psk_demodulate(rx, bits)
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics():
    rows = []
    for station in STATIONS:
        fp = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
        xp = os.path.join(PROCESSED_PATH, f"{station}_features.csv")

        sources = []
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            cols = [c for c in SIGNAL_COLS_FILTERED.get(station, []) if c in df.columns]
            sources.append(("filtered", df, cols))
        if os.path.exists(xp):
            df = pd.read_csv(xp)
            cols = [c for c in SIGNAL_COLS_FEATURES.get(station, []) if c in df.columns]
            sources.append(("features", df, cols))

        for src, df, cols in sources:
            for col in cols:
                x = df[col].fillna(0).to_numpy(dtype=float)
                if len(x) < 10:
                    continue
                _, _, _, snr_am           = pipeline_am(x)
                _, _, _, snr_fm           = pipeline_fm(x)
                _, _, _, _, snr_ask, ber_ask, _ = pipeline_ask(x)
                _, _, _, snr_fsk, ber_fsk, _    = pipeline_fsk(x)
                _, _, _, snr_psk, ber_psk, _    = pipeline_psk(x)

                rows.append({
                    "station": station, "source": src, "signal_col": col,
                    "AM_SNR_dB" : round(snr_am,  3),
                    "FM_SNR_dB" : round(snr_fm,  3),
                    "ASK_SNR_dB": round(snr_ask, 3),
                    "FSK_SNR_dB": round(snr_fsk, 3),
                    "PSK_SNR_dB": round(snr_psk, 3),
                    "ASK_BER"   : round(ber_ask, 5),
                    "FSK_BER"   : round(ber_fsk, 5),
                    "PSK_BER"   : round(ber_psk, 5),
                })
    return pd.DataFrame(rows)


def main():
    print("=" * 65)
    print("TELE 523 · Demodulation metrics (Student 3)")
    print("=" * 65)
    df = compute_metrics()
    out = os.path.join(OUT_DIR, "demodulation_metrics.csv")
    df.to_csv(out, index=False)
    print(f"Metrics saved -> {out}")
    print("\nSNR summary (mean per station x scheme):")
    print(df.pivot_table(values=["AM_SNR_dB","FM_SNR_dB","ASK_SNR_dB","FSK_SNR_dB","PSK_SNR_dB"],
                         index="station", aggfunc="mean").round(2).to_string())
    print("\nBER summary:")
    print(df.pivot_table(values=["ASK_BER","FSK_BER","PSK_BER"],
                         index="station", aggfunc="mean").round(5).to_string())


if __name__ == "__main__":
    main()