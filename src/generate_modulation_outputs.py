"""
generate_modulation_outputs_v2.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Each station's filtered signal columns pass through ALL 5 modulation schemes
independently. Output: one CSV per scheme per station (35 files total).

Structure:
  {STATION}_AM.csv   — columns: timestamp, {col}_baseband, {col}_am_demodulated, ...
  {STATION}_FM.csv   — columns: timestamp, {col}_baseband, {col}_fm_demodulated, ...
  {STATION}_ASK.csv  — columns: timestamp, {col}_baseband, {col}_ask_recovered, ...
  {STATION}_FSK.csv  — columns: timestamp, {col}_baseband, {col}_fsk_recovered, ...
  {STATION}_PSK.csv  — columns: timestamp, {col}_baseband, {col}_psk_recovered, ...

Each file also carries current_state_binary for Student 4's labelling.
Analog demodulated outputs (AM, FM, ASK envelope) are normalised to [0,1]
so Student 4 can apply quantization directly.
Digital recovered outputs (ASK bits, FSK bits, PSK bits) are 0/1 integers
ready for line coding (NRZ, Manchester) and PCM.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert

PROCESSED_PATH = "/home/claude/data/processed"
OUT_DIR        = "/home/claude/results/modulation/output_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters
FS        = 0.5
FC        = 0.05
NOISE_STD = 0.05
F0_FSK    = 0.03
F1_FSK    = 0.07
MOD_INDEX = 0.8
KF        = 0.1

SIGNAL_COLS = {
    "EC_1"  : ["i3_photoresistor", "current_task_duration"],
    "HBW_1" : ["m1_speed", "m2_speed", "m3_speed", "m4_speed", "current_task_duration"],
    "MM_1"  : ["m1_speed", "m2_speed", "m3_speed", "current_task_duration"],
    "OV_1"  : ["m1_speed", "current_task_duration"],
    "SM_1"  : ["m1_speed", "current_task_duration"],
    "VGR_1" : ["m1_speed", "m2_speed", "m3_speed", "current_task_duration"],
    "WT_1"  : ["m2_speed", "current_task_duration"],
}

# ── Utilities ──────────────────────────────────────────────────────────────────
def t_ax(n):     return np.arange(n) / FS
def norm(s):
    lo, hi = s.min(), s.max()
    return np.zeros_like(s, dtype=float) if hi-lo < 1e-12 else (s-lo)/(hi-lo)
def awgn(s, seed=42):
    return s + np.random.default_rng(seed).normal(0, NOISE_STD, len(s))
def s2bits(s):   return (np.clip(norm(s), 0, 1) > 0.5).astype(int)
def spb(n, nb):  return max(1, n // max(1, nb))

# ── AM ─────────────────────────────────────────────────────────────────────────
def am_pipeline(x):
    t = t_ax(len(x))
    mod = (1 + MOD_INDEX * norm(x)) * np.cos(2*np.pi*FC*t)
    rx  = awgn(mod)
    env = np.abs(hilbert(rx)); env -= env.mean()
    return np.clip(norm(env), 0, 1)          # normalised demodulated signal

# ── FM ─────────────────────────────────────────────────────────────────────────
def fm_pipeline(x):
    t = t_ax(len(x))
    mod = np.cos(2*np.pi*FC*t + 2*np.pi*KF*np.cumsum(norm(x))/FS)
    rx  = awgn(mod)
    phase = np.unwrap(np.angle(hilbert(rx)))
    ifreq = np.append(np.diff(phase)/(2*np.pi/FS), 0) - FC
    return np.clip(norm(ifreq), 0, 1)

# ── ASK ────────────────────────────────────────────────────────────────────────
def ask_pipeline(x):
    n    = len(x)
    bits = s2bits(x)
    nb   = len(bits)
    s    = spb(n, nb)
    t    = t_ax(n)
    env  = np.repeat(bits.astype(float), s)[:n]
    if len(env) < n: env = np.pad(env, (0, n-len(env)), constant_values=env[-1])
    mod  = env * np.cos(2*np.pi*FC*t)
    rx   = awgn(mod)
    # Envelope detection → recovered bit per sample window
    env_rx    = np.abs(hilbert(rx))
    rec_bits  = np.array([1 if np.mean(env_rx[i*s:(i+1)*s]) > 0.5 else 0
                          for i in range(nb)])
    # Expand recovered bits back to sample resolution
    recovered = np.repeat(rec_bits, s)[:n]
    if len(recovered) < n: recovered = np.pad(recovered, (0, n-len(recovered)), constant_values=recovered[-1])
    return recovered.astype(int), bits, rec_bits   # sample-res, original bits, recovered bits

# ── FSK ────────────────────────────────────────────────────────────────────────
def fsk_pipeline(x):
    n    = len(x)
    bits = s2bits(x)
    nb   = len(bits)
    s    = spb(n, nb)
    t    = t_ax(n)
    mod  = np.zeros(n)
    for i, b in enumerate(bits):
        st, en = i*s, min((i+1)*s, n)
        mod[st:en] = np.cos(2*np.pi*(F1_FSK if b else F0_FSK)*t[st:en])
    rx   = awgn(mod)
    r0, r1 = np.cos(2*np.pi*F0_FSK*t), np.cos(2*np.pi*F1_FSK*t)
    rec_bits = np.array([
        1 if abs(np.dot(rx[i*s:min((i+1)*s,n)], r1[i*s:min((i+1)*s,n)])) >
             abs(np.dot(rx[i*s:min((i+1)*s,n)], r0[i*s:min((i+1)*s,n)])) else 0
        for i in range(nb)])
    recovered = np.repeat(rec_bits, s)[:n]
    if len(recovered) < n: recovered = np.pad(recovered, (0, n-len(recovered)), constant_values=recovered[-1])
    return recovered.astype(int), bits, rec_bits

# ── PSK (BPSK) ─────────────────────────────────────────────────────────────────
def psk_pipeline(x):
    n    = len(x)
    bits = s2bits(x)
    nb   = len(bits)
    s    = spb(n, nb)
    t    = t_ax(n)
    phases = np.repeat(np.where(bits==1, 0, np.pi), s)[:n]
    if len(phases) < n: phases = np.pad(phases, (0, n-len(phases)), constant_values=phases[-1])
    mod  = np.cos(2*np.pi*FC*t + phases)
    rx   = awgn(mod)
    prod = rx * np.cos(2*np.pi*FC*t)
    rec_bits = np.array([1 if np.mean(prod[i*s:min((i+1)*s,n)]) > 0 else 0
                         for i in range(nb)])
    recovered = np.repeat(rec_bits, s)[:n]
    if len(recovered) < n: recovered = np.pad(recovered, (0, n-len(recovered)), constant_values=recovered[-1])
    return recovered.astype(int), bits, rec_bits

# ══════════════════════════════════════════════════════════════════════════════
SCHEMES = ["AM", "FM", "ASK", "FSK", "PSK"]

for station in sorted(SIGNAL_COLS.keys()):
    fpath = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    if not os.path.exists(fpath):
        print(f"[SKIP] {station}"); continue

    df      = pd.read_csv(fpath, parse_dates=["timestamp"])
    cols    = [c for c in SIGNAL_COLS[station] if c in df.columns]
    labels  = df["current_state_binary"].fillna(0).astype(int) if "current_state_binary" in df.columns else pd.Series(np.zeros(len(df), dtype=int))
    ts      = df["timestamp"]

    # Build one output dict per scheme
    out = {s: {"timestamp": ts, "current_state_binary": labels} for s in SCHEMES}

    for col in cols:
        x = df[col].fillna(0).to_numpy(dtype=float)
        if len(x) < 30: continue

        # Every signal column passes through every scheme
        out["AM"][f"{col}_baseband"]      = np.round(norm(x), 6)
        out["AM"][f"{col}_am_demodulated"]= np.round(am_pipeline(x), 6)

        out["FM"][f"{col}_baseband"]      = np.round(norm(x), 6)
        out["FM"][f"{col}_fm_demodulated"]= np.round(fm_pipeline(x), 6)

        ask_rec, ask_orig, _ = ask_pipeline(x)
        out["ASK"][f"{col}_baseband"]     = np.round(norm(x), 6)
        out["ASK"][f"{col}_original_bits"]= ask_orig
        out["ASK"][f"{col}_ask_recovered"]= ask_rec

        fsk_rec, fsk_orig, _ = fsk_pipeline(x)
        out["FSK"][f"{col}_baseband"]     = np.round(norm(x), 6)
        out["FSK"][f"{col}_original_bits"]= fsk_orig
        out["FSK"][f"{col}_fsk_recovered"]= fsk_rec

        psk_rec, psk_orig, _ = psk_pipeline(x)
        out["PSK"][f"{col}_baseband"]     = np.round(norm(x), 6)
        out["PSK"][f"{col}_original_bits"]= psk_orig
        out["PSK"][f"{col}_psk_recovered"]= psk_rec

    for scheme in SCHEMES:
        df_out = pd.DataFrame(out[scheme])
        # Put timestamp and label at the edges
        cols_order = (["timestamp"] +
                      [c for c in df_out.columns if c not in ("timestamp","current_state_binary")] +
                      ["current_state_binary"])
        df_out = df_out[cols_order]
        path = os.path.join(OUT_DIR, f"{station}_{scheme}.csv")
        df_out.to_csv(path, index=False)
        signal_data_cols = [c for c in df_out.columns if c not in ("timestamp","current_state_binary")]
        print(f"  {station}_{scheme}.csv  — {len(df_out)} rows, signal cols: {signal_data_cols}")

print(f"\nDone — {7*5} output files in {OUT_DIR}")