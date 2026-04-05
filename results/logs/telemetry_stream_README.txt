TELEMETRY STREAM LOG — MONITORING LEAD GUIDE
TELE 523 · Group 1 · Digital Telemetry Lead Output
============================================================

FILE
  telemetry_stream.log

FORMAT
  JSONL (Newline-Delimited JSON) — one JSON object per line.
  Each line = one telemetry frame = one 60-second segment from
  one station, processed through the full digital telemetry chain:
  quantization → PCM encoding → line coding → channel simulation → decode.

  Total frames    : 875
  Simulated span  : 2026-03-31T09:00:00  →  2026-03-31T16:17:00
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
the segment, keyed as  {signal}_{feature}_{source}:

  Sources  : _baseband  |  _am_demodulated
  Features : _rms  _mean  _variance  _fft_peak
             _psd_peak  _psd_mean_energy  _psd_peak_freq

Examples (MM_1):
  frame["features"]["m1_speed_rms_baseband"]
  frame["features"]["m1_speed_psd_peak_am_demodulated"]
  frame["features"]["m1_speed_psd_peak_freq_baseband"]

To pull all RMS features from a frame:
  rms = {k: v for k,v in frame["features"].items() if "_rms_" in k}

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
          st.error(f"RED  {frame['station']}: {alert}")
      elif "INTEGRITY" in alert or "CHECKSUM" in alert:
          st.warning(f"WARN {frame['station']}: {alert}")
      elif "BER_WARN" in alert or "PARITY" in alert:
          st.info(f"INFO {frame['station']}: {alert}")


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
