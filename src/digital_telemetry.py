"""
generate_telemetry_log.py
TELE 523 · Group 1 — Digital Telemetry Lead
───────────────────────────────────────────────────────────────
Converts all {STATION}_AM_monitoring_ready.csv files into a
single JSONL telemetry stream log the Monitoring Lead can
tail / stream into their Streamlit dashboard.

Log format (one JSON object per line):
  {
    "log_ts"               : "2023-02-06 14:08:08",
    "recv_ts"              : "2026-03-31T09:00:00",
    "station"              : "MM_1",
    "mod_scheme"           : "AM",
    "segment_idx"          : 0,
    "segment_start"        : "2023-02-06T14:08:08",
    "segment_end"          : "2023-02-06T14:09:06",
    "machine_state"        : 1,
    "dt_integrity_flag"    : 1,
    "dt_parity_pass_rate"  : 0.972,
    "dt_checksum_ok_rate"  : 1.0,
    "dt_best_bit_depth"    : 16,
    "dt_channel_ber_awgn"  : 0.000528,
    "dt_channel_ber_fading": 0.501,
    "features"             : { ... all signal features ... },
    "reconstructed"        : { ... PCM-decoded segment stats ... },
    "alerts"               : []
  }
"""

import os, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROC_DIR         = "data/raw"
OUT_DIR          = "results/logs"
STATIONS         = ["EC_1", "HBW_1", "MM_1", "OV_1", "SM_1", "VGR_1", "WT_1"]
MOD_SCHEME       = "AM"
SIM_START        = datetime(2026, 3, 31, 9, 0, 0)
SIM_INTERVAL_SEC = 30

THRESH_PARITY    = 0.95
THRESH_CS        = 0.90
THRESH_BER_AWGN  = 0.01


def _feature_cols(df):
    skip = {
        "segment_idx","segment_start","segment_end","current_state_binary",
        "dt_parity_pass_rate","dt_checksum_ok_rate","dt_best_bit_depth",
        "dt_mod_scheme","dt_channel_ber_awgn","dt_channel_ber_fading","dt_integrity_flag",
    }
    recon = ("_reconstructed_mean", "_reconstructed_rms")
    return [c for c in df.columns if c not in skip and not any(c.endswith(s) for s in recon)]


def _recon_cols(df):
    return [c for c in df.columns
            if c.endswith("_reconstructed_mean") or c.endswith("_reconstructed_rms")]


def _alerts(row):
    a = []
    if row.get("dt_integrity_flag", 1) == 0:
        a.append("INTEGRITY_WARN: telemetry integrity flag degraded")
    if row.get("dt_parity_pass_rate", 1.0) < THRESH_PARITY:
        a.append(f"PARITY_WARN: pass_rate={row['dt_parity_pass_rate']:.3f} < {THRESH_PARITY}")
    if row.get("dt_checksum_ok_rate", 1.0) < THRESH_CS:
        a.append(f"CHECKSUM_WARN: ok_rate={row['dt_checksum_ok_rate']:.3f} < {THRESH_CS}")
    if row.get("dt_channel_ber_awgn", 0.0) > THRESH_BER_AWGN:
        a.append(f"BER_WARN: ber_awgn={row['dt_channel_ber_awgn']:.6f} > {THRESH_BER_AWGN}")
    if row.get("current_state_binary", 1) == 0:
        a.append("MACHINE_FAULT: station not in ready state")
    return a


def _safe(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 6)
    return v


def generate_log():
    os.makedirs(OUT_DIR, exist_ok=True)

    log_path    = os.path.join(OUT_DIR, "telemetry_stream.log")
    readme_path = os.path.join(OUT_DIR, "telemetry_stream_README.txt")

    all_frames = []

    for station in STATIONS:
        csv_path = os.path.join(PROC_DIR, f"{station}_features_{MOD_SCHEME}_demod.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {station}")
            continue

        df = pd.read_csv(csv_path)
        feat_cols  = _feature_cols(df)
        recon_cols = _recon_cols(df)
        print(f"  {station}: {len(df)} segments | {len(feat_cols)} features | {len(recon_cols)} recon cols")

        for _, row in df.iterrows():
            station_offset = STATIONS.index(station)
            recv_ts = SIM_START + timedelta(
                seconds=(int(row["segment_idx"]) * len(STATIONS) + station_offset) * SIM_INTERVAL_SEC
            )

            features     = {c: _safe(row[c]) for c in feat_cols}
            reconstructed = {c: _safe(row[c]) for c in recon_cols}

            frame = {
                "log_ts"               : str(row["segment_start"])[:19].replace("T", " "),
                "recv_ts"              : recv_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "station"              : station,
                "mod_scheme"           : MOD_SCHEME,
                "segment_idx"          : int(row["segment_idx"]),
                "segment_start"        : str(row["segment_start"])[:19],
                "segment_end"          : str(row["segment_end"])[:19],
                "machine_state"        : int(row["current_state_binary"]),
                "dt_integrity_flag"    : int(row.get("dt_integrity_flag", 1)),
                "dt_parity_pass_rate"  : round(float(row.get("dt_parity_pass_rate", 1.0)), 4),
                "dt_checksum_ok_rate"  : round(float(row.get("dt_checksum_ok_rate", 1.0)),  4),
                "dt_best_bit_depth"    : int(row.get("dt_best_bit_depth", 8)),
                "dt_channel_ber_awgn"  : round(float(row.get("dt_channel_ber_awgn", 0.0)),  6),
                "dt_channel_ber_fading": round(float(row.get("dt_channel_ber_fading", 0.0)), 6),
                "features"             : features,
                "reconstructed"        : reconstructed,
                "alerts"               : _alerts(row),
            }
            all_frames.append(frame)

    all_frames.sort(key=lambda f: (f["recv_ts"], f["station"]))

    with open(log_path, "w", encoding="utf-8") as fout:
        for frame in all_frames:
            fout.write(json.dumps(frame) + "\n")

    total_alerts = sum(len(f["alerts"]) for f in all_frames)
    print(f"\n  Log written  → {log_path}")
    print(f"  Frames       : {len(all_frames)}")
    print(f"  Alerts fired : {total_alerts}")
    print(f"  Time span    : {all_frames[0]['recv_ts']} → {all_frames[-1]['recv_ts']}")

    _write_readme(readme_path, len(all_frames),
                  all_frames[0]["recv_ts"], all_frames[-1]["recv_ts"])


def _write_readme(path, n_frames, t_start, t_end):
    txt = f"""TELEMETRY STREAM LOG — MONITORING LEAD GUIDE
TELE 523 · Group 1 · Digital Telemetry Lead Output
============================================================

FILE
  telemetry_stream.log

FORMAT
  JSONL (Newline-Delimited JSON) — one JSON object per line.
  Each line = one telemetry frame = one 60-second segment from
  one station, processed through the full digital telemetry chain:
  quantization → PCM encoding → line coding → channel simulation → decode.

  Total frames    : {n_frames}
  Simulated span  : {t_start}  →  {t_end}
  Frame interval  : 30 seconds between consecutive frames
  Stations        : EC_1  HBW_1  MM_1  OV_1  SM_1  VGR_1  WT_1


HOW TO READ THE LOG IN PYTHON
──────────────────────────────────────────────────────────────

  # Option A — load all at once (batch / replay mode)
  import json
  frames = []
  with open("results/logs/telemetry_stream.log") as f:
      for line in f:
          frames.append(json.loads(line.strip()))

  # Option B — stream line by line (simulate live feed)
  import json, time
  with open("results/logs/telemetry_stream.log") as f:
      for line in f:
          frame = json.loads(line.strip())
          process(frame)     # your Streamlit update function
          time.sleep(0.5)    # adjust for replay speed


FRAME KEYS
──────────────────────────────────────────────────────────────
  log_ts               Original factory timestamp (string)
  recv_ts              Simulated receive timestamp — USE FOR X-AXIS (string)
  station              Station ID  e.g. "MM_1"
  mod_scheme           Demodulation scheme  "AM"
  segment_idx          Segment number 0-124
  segment_start        Start of 60-second window
  segment_end          End of 60-second window
  machine_state        1 = READY  |  0 = FAULT
  dt_integrity_flag    1 = OK     |  0 = DEGRADED
  dt_parity_pass_rate  Fraction of 8-bit parity blocks passed  [0,1]
  dt_checksum_ok_rate  Fraction of frames with matching 16-bit checksum  [0,1]
  dt_best_bit_depth    PCM bit depth used  (16 for all stations)
  dt_channel_ber_awgn  BER at 10 dB SNR under AWGN
  dt_channel_ber_fading  BER at 10 dB SNR under AWGN + Rayleigh fading
  features             dict — all signal features (see below)
  reconstructed        dict — PCM-decoded signal stats (see below)
  alerts               list — active alert strings (empty = no alerts)


FEATURE EXTRACTION
──────────────────────────────────────────────────────────────
The "features" dict contains all Phase-2 signal features for
the segment, keyed as  {{signal}}_{{feature}}_{{source}}:

  Sources  : _baseband  |  _am_demodulated
  Features : _rms  _mean  _variance  _fft_peak
             _psd_peak  _psd_mean_energy  _psd_peak_freq

Examples (MM_1):
  frame["features"]["m1_speed_rms_baseband"]
  frame["features"]["m1_speed_psd_peak_am_demodulated"]
  frame["features"]["m1_speed_psd_peak_freq_baseband"]

To pull all RMS features from a frame:
  rms = {{k: v for k,v in frame["features"].items() if "_rms_" in k}}

To compare baseband vs demodulated for one signal:
  bb    = frame["features"]["m1_speed_rms_baseband"]
  demod = frame["features"]["m1_speed_rms_am_demodulated"]

The "reconstructed" dict contains PCM-decoded per-segment stats:
  frame["reconstructed"]["m1_speed_baseband_reconstructed_mean"]
  frame["reconstructed"]["m1_speed_baseband_reconstructed_rms"]


ALERT LOGIC
──────────────────────────────────────────────────────────────
  for alert in frame["alerts"]:
      if "MACHINE_FAULT" in alert:
          st.error(f"RED  {{frame['station']}}: {{alert}}")
      elif "INTEGRITY" in alert or "CHECKSUM" in alert:
          st.warning(f"WARN {{frame['station']}}: {{alert}}")
      elif "BER_WARN" in alert or "PARITY" in alert:
          st.info(f"INFO {{frame['station']}}: {{alert}}")


SUGGESTED STREAMLIT LAYOUT
──────────────────────────────────────────────────────────────
  Sidebar
    station       multiselect (EC_1 ... WT_1)
    replay_speed  slider 0.1x → 5x (controls time.sleep)
    feature       selectbox (pick which feature to plot)

  Tab 1 — Live Feed
    st.metric  machine_state | dt_integrity_flag | dt_channel_ber_awgn
    st.dataframe  last 20 frames for selected station
    Alert banner  if frame["alerts"]: st.error(...)

  Tab 2 — Feature Trends
    st.line_chart  chosen feature over recv_ts
    Overlay baseband vs am_demodulated vs reconstructed_mean

  Tab 3 — Channel Health
    Bar chart  dt_channel_ber_awgn vs dt_channel_ber_fading per station
    Progress bars  dt_parity_pass_rate | dt_checksum_ok_rate

  Tab 4 — PSD Analysis
    Bar chart  psd_peak and psd_mean_energy per signal
    Scatter    psd_peak_freq per signal

  Tab 5 — Station Map
    Heatmap of any feature across all 7 stations over time


REPLAY SPEED REFERENCE
──────────────────────────────────────────────────────────────
  Real time   time.sleep(30)   1 frame per 30 seconds
  Fast        time.sleep(1)    1 frame per second
  Demo        time.sleep(0.2)  5 frames per second
  Instant     time.sleep(0)    plot everything statically

============================================================
Digital Telemetry Lead · TELE 523 Group 1
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"  README written → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  TELE 523 · Telemetry Log Generator")
    print("=" * 60)
    generate_log()
    print("=" * 60)