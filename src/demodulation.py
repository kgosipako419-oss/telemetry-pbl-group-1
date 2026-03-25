"""
demodulation.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Standalone demodulation module. Receives modulated + channel-corrupted signals
and recovers the original baseband signal or bit sequence.

This module is the explicit Channel → Demodulation stage of the pipeline:

  Baseband → [modulation.py] → Modulated → [AWGN channel] → Received
          → [demodulation.py] → Recovered signal / bits
          → [generate_demodulation_outputs.py] → CSV files for Student 4

All demodulation functions mirror their corresponding modulation schemes
in modulation.py. They can be used standalone or imported by
generate_demodulation_outputs.py.

Demodulation methods:
  AM  — Hilbert envelope detection, DC removal, normalised to [0, 1]
  FM  — Instantaneous frequency via phase differentiation, normalised to [0, 1]
  ASK — Envelope detection + per-bit-window threshold → {0, 1} bits
  FSK — Matched filter correlation (f0 vs f1 reference tones) → {0, 1} bits
  PSK — Coherent detection (multiply by reference carrier, integrate) → {0, 1} bits
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# ── Import shared parameters and utilities from modulation.py ─────────────────
try:
    from modulation import (
        FS, FC, NOISE_STD, F0_FSK, F1_FSK,
        SIGNAL_COLS,
        normalise, _expand_bits, _spb,
        awgn, signal_to_bits,
        snr_db, ber,
        am_modulate, fm_modulate, ask_modulate, fsk_modulate, psk_modulate,
        channel,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modulation import (
        FS, FC, NOISE_STD, F0_FSK, F1_FSK,
        SIGNAL_COLS,
        normalise, _expand_bits, _spb,
        awgn, signal_to_bits,
        snr_db, ber,
        am_modulate, fm_modulate, ask_modulate, fsk_modulate, psk_modulate,
        channel,
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR        = os.path.join(BASE_DIR, "results", "demodulation")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 — AM DEMODULATION
# =============================================================================

def demod_am(received):
    """
    AM demodulation — Hilbert envelope detection.

    Steps:
      1. Compute the analytic signal via Hilbert transform
      2. Take the magnitude → envelope
      3. Remove DC offset introduced by the carrier
      4. Normalise to [0, 1] so Student 4 can quantize directly

    Input:  received  — AM modulated signal after AWGN channel  (float array)
    Output: envelope  — recovered analog signal, normalised [0, 1]
    """
    env = np.abs(hilbert(received))
    env -= env.mean()
    return np.clip(normalise(env), 0, 1)


# =============================================================================
# SECTION 2 — FM DEMODULATION
# =============================================================================

def demod_fm(received, fc=FC, fs=FS):
    """
    FM demodulation — instantaneous frequency via phase differentiation.

    Steps:
      1. Compute the analytic signal via Hilbert transform
      2. Unwrap the instantaneous phase
      3. Differentiate phase to get instantaneous frequency
      4. Subtract carrier offset fc to recover the message
      5. Normalise to [0, 1] so Student 4 can quantize directly

    Input:  received  — FM modulated signal after AWGN channel  (float array)
    Output: recovered — recovered analog signal, normalised [0, 1]
    """
    phase     = np.unwrap(np.angle(hilbert(received)))
    inst_freq = np.diff(phase) / (2 * np.pi / fs)
    inst_freq = np.append(inst_freq, inst_freq[-1]) - fc
    return np.clip(normalise(inst_freq), 0, 1)


# =============================================================================
# SECTION 3 — ASK DEMODULATION
# =============================================================================

def demod_ask(received, n_bits):
    """
    ASK demodulation — envelope detection + per-bit-window threshold.

    Steps:
      1. Compute envelope via Hilbert magnitude
      2. Divide signal into n_bits windows of equal length
      3. Take mean of envelope in each window
      4. Threshold at 0.5: mean > 0.5 → bit=1, else → bit=0

    Input:
      received  — ASK modulated signal after AWGN channel  (float array)
      n_bits    — number of bits to recover (must match original bit count)

    Output:
      recovered_bits  — 0/1 integer array, length = n_bits
      envelope        — continuous envelope, normalised [0, 1], length = len(received)
    """
    n   = len(received)
    s   = _spb(n, n_bits)
    env = np.abs(hilbert(received))
    recovered = np.array([
        1 if np.mean(env[i*s : (i+1)*s]) > 0.5 else 0
        for i in range(n_bits)
    ], dtype=int)
    return recovered, np.clip(normalise(env), 0, 1)


# =============================================================================
# SECTION 4 — FSK DEMODULATION
# =============================================================================

def demod_fsk(received, n_bits, f0=F0_FSK, f1=F1_FSK, fs=FS):
    """
    FSK demodulation — matched filter (correlation) detection.

    Steps:
      1. Build reference cosine tones at f0 and f1 for the full signal length
      2. Divide signal into n_bits windows of equal length
      3. In each window, compute dot-product with both reference tones
      4. Assign bit=1 if |corr(f1)| > |corr(f0)|, else bit=0

    Input:
      received  — FSK modulated signal after AWGN channel  (float array)
      n_bits    — number of bits to recover
      f0, f1    — FSK mark/space frequencies (must match modulator)

    Output:
      recovered_bits — 0/1 integer array, length = n_bits
    """
    n   = len(received)
    s   = _spb(n, n_bits)
    t   = np.arange(n) / fs
    r0  = np.cos(2 * np.pi * f0 * t)
    r1  = np.cos(2 * np.pi * f1 * t)
    recovered = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        st, en = i*s, min((i+1)*s, n)
        c0 = abs(np.dot(received[st:en], r0[st:en]))
        c1 = abs(np.dot(received[st:en], r1[st:en]))
        recovered[i] = 1 if c1 > c0 else 0
    return recovered


# =============================================================================
# SECTION 5 — PSK DEMODULATION
# =============================================================================

def demod_psk(received, n_bits, fc=FC, fs=FS):
    """
    BPSK demodulation — coherent detection.

    Steps:
      1. Multiply received signal by reference carrier cos(2*pi*fc*t)
      2. Divide into n_bits windows
      3. Integrate (take mean) over each window
      4. Positive mean → bit=1, negative mean → bit=0

    Input:
      received  — PSK modulated signal after AWGN channel  (float array)
      n_bits    — number of bits to recover

    Output:
      recovered_bits — 0/1 integer array, length = n_bits
    """
    n   = len(received)
    s   = _spb(n, n_bits)
    t   = np.arange(n) / fs
    ref = np.cos(2 * np.pi * fc * t)
    product = received * ref
    recovered = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        st, en = i*s, min((i+1)*s, n)
        recovered[i] = 1 if np.mean(product[st:en]) > 0 else 0
    return recovered


# =============================================================================
# SECTION 6 — FULL DEMODULATION PIPELINES PER SCHEME
# Each function: takes baseband → modulates → applies channel → demodulates
# Returns everything needed for output files and metrics
# =============================================================================

def pipeline_am(baseband):
    """
    Full AM pipeline: baseband → modulate → channel → demodulate.

    Returns:
      modulated   — clean AM signal before channel
      received    — noisy signal after AWGN channel
      demodulated — recovered analog signal, normalised [0, 1]
      snr         — SNR in dB between modulated and received
    """
    mod = am_modulate(baseband)
    rx  = channel(mod)
    dem = demod_am(rx)
    return mod, rx, dem, snr_db(mod, rx)


def pipeline_fm(baseband):
    """
    Full FM pipeline: baseband → modulate → channel → demodulate.

    Returns:
      modulated   — clean FM signal before channel
      received    — noisy signal after AWGN channel
      demodulated — recovered analog signal, normalised [0, 1]
      snr         — SNR in dB between modulated and received
    """
    mod = fm_modulate(baseband)
    rx  = channel(mod)
    dem = demod_fm(rx)
    return mod, rx, dem, snr_db(mod, rx)


def pipeline_ask(baseband):
    """
    Full ASK pipeline: baseband → bits → modulate → channel → demodulate.

    Returns:
      modulated       — clean ASK signal before channel
      received        — noisy signal after AWGN channel
      recovered_bits  — demodulated bit array {0,1}
      envelope        — continuous demodulated envelope [0,1]
      snr             — SNR in dB
      ber_val         — Bit Error Rate vs original bits
      original_bits   — bit array derived from baseband before modulation
    """
    bits     = signal_to_bits(baseband)
    mod      = ask_modulate(bits, len(baseband))
    rx       = channel(mod)
    rec, env = demod_ask(rx, len(bits))
    return mod, rx, rec, env, snr_db(mod, rx), ber(bits, rec), bits


def pipeline_fsk(baseband):
    """
    Full FSK pipeline: baseband → bits → modulate → channel → demodulate.

    Returns:
      modulated       — clean FSK signal before channel
      received        — noisy signal after AWGN channel
      recovered_bits  — demodulated bit array {0,1}
      snr             — SNR in dB
      ber_val         — Bit Error Rate vs original bits
      original_bits   — bit array derived from baseband before modulation
    """
    bits = signal_to_bits(baseband)
    mod  = fsk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = demod_fsk(rx, len(bits))
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits


def pipeline_psk(baseband):
    """
    Full PSK pipeline: baseband → bits → modulate → channel → demodulate.

    Returns:
      modulated       — clean PSK signal before channel
      received        — noisy signal after AWGN channel
      recovered_bits  — demodulated bit array {0,1}
      snr             — SNR in dB
      ber_val         — Bit Error Rate vs original bits
      original_bits   — bit array derived from baseband before modulation
    """
    bits = signal_to_bits(baseband)
    mod  = psk_modulate(bits, len(baseband))
    rx   = channel(mod)
    rec  = demod_psk(rx, len(bits))
    return mod, rx, rec, snr_db(mod, rx), ber(bits, rec), bits


# =============================================================================
# SECTION 7 — METRICS SUMMARY
# Compute SNR and BER for all stations x schemes x signal columns
# =============================================================================

def compute_metrics():
    """
    Run every station × scheme × signal column through the full pipeline
    and return a DataFrame of SNR and BER metrics.
    """
    rows = []
    for station in sorted(SIGNAL_COLS.keys()):
        fpath = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
        if not os.path.exists(fpath):
            continue
        df       = pd.read_csv(fpath, parse_dates=["timestamp"])
        sig_cols = [c for c in SIGNAL_COLS[station] if c in df.columns]

        for col in sig_cols:
            x = df[col].fillna(0).to_numpy(dtype=float)
            if len(x) < 30:
                continue

            _, _, _, snr_am           = pipeline_am(x)
            _, _, _, snr_fm           = pipeline_fm(x)
            _, _, _, _, snr_ask, ber_ask, _ = pipeline_ask(x)
            _, _, _, snr_fsk, ber_fsk, _    = pipeline_fsk(x)
            _, _, _, snr_psk, ber_psk, _    = pipeline_psk(x)

            rows.append({
                "station"    : station,
                "signal_col" : col,
                "AM_SNR_dB"  : round(snr_am,  3),
                "FM_SNR_dB"  : round(snr_fm,  3),
                "ASK_SNR_dB" : round(snr_ask, 3),
                "FSK_SNR_dB" : round(snr_fsk, 3),
                "PSK_SNR_dB" : round(snr_psk, 3),
                "ASK_BER"    : round(ber_ask, 5),
                "FSK_BER"    : round(ber_fsk, 5),
                "PSK_BER"    : round(ber_psk, 5),
            })

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 8 — MAIN  (run standalone for metrics only)
# =============================================================================

def main():
    print("=" * 65)
    print("TELE 523 · Demodulation module (Student 3)")
    print("Run generate_demodulation_outputs.py to produce handoff CSVs")
    print("=" * 65)
    print("\nComputing SNR / BER metrics for all stations x schemes...\n")

    df = compute_metrics()

    out_csv = os.path.join(OUT_DIR, "demodulation_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Metrics saved → {out_csv}\n")

    print("SNR (dB) summary:")
    pivot = df.pivot_table(
        values=["AM_SNR_dB", "FM_SNR_dB", "ASK_SNR_dB", "FSK_SNR_dB", "PSK_SNR_dB"],
        index="station", aggfunc="mean"
    ).round(2)
    print(pivot.to_string())

    print("\nBER summary (digital schemes):")
    pivot_ber = df.pivot_table(
        values=["ASK_BER", "FSK_BER", "PSK_BER"],
        index="station", aggfunc="mean"
    ).round(5)
    print(pivot_ber.to_string())


if __name__ == "__main__":
    main()