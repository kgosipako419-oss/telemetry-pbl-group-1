TELEMETRY STREAM LOG — MONITORING LEAD GUIDE
TELE 523 · Group 1 · Digital Telemetry Lead Output
============================================================

WHY PSK AND ONLY PSK?
──────────────────────────────────────────────────────────────
The Digital Telemetry Lead evaluated all five modulation schemes
(AM, FM, ASK, FSK, PSK) against the same underlying sensor data
from all seven stations. The key findings were:

  1. THE UNDERLYING INFORMATION IS IDENTICAL ACROSS ALL SCHEMES.
     Every CSV file for the same station and segment contains the
     exact same baseband signal values. The modulation scheme
     only determines HOW that information was carried over the
     channel — not what the information is.

  2. THE SCHEMES DIFFER IN TWO IMPORTANT WAYS:

     a) Signal type of the recovered output:
        • AM and FM  →  continuous analogue reconstruction.
          The recovered values are floating-point numbers that
          approximate the original waveform shape.
        • ASK, FSK, PSK  →  binary bitstream recovery.
          The recovered values are 0 or 1 only. These schemes
          are digital — they do not attempt to reconstruct a
          waveform but instead recover transmitted bits.

     b) Recovery fidelity (correlation of recovered vs baseband):
        Scheme   Avg Corr   Avg RMSE   Signal type
        PSK      0.87       0.18       Digital (binary)
        AM       0.69       0.23       Analogue (continuous)
        ASK      0.67       0.29       Digital (binary)
        FM       0.54       0.27       Analogue (continuous)
        FSK      0.02       0.63       Digital (binary)  ← worst

  3. PSK IS THE BEST SCHEME because:
     • Highest correlation with the original baseband signal (0.87)
       — measured consistently across all seven stations.
     • Lowest theoretical BER under AWGN at 10 dB SNR (0.000006),
       orders of magnitude better than AM (0.001) and FM (0.003).
     • Lower Rayleigh fading BER (0.044) than AM (0.023 is similar
       but PSK’s digital nature gives more robust decoding).
     • Phase-shift keying encodes information in the phase of the
       carrier, making it immune to amplitude noise — the dominant
       interference in this factory telemetry environment.
     • FSK is clearly the worst: near-zero correlation (∼0.02)
       across all stations, meaning its recovered bitstream is
       essentially decorrelated from the source signal.

  4. THE COMPARISON DATA IS PRESERVED.
     The full five-modulation analysis (SQNR, BER, parity, line
     coding) is retained in digital_telemetry_report.json and the
     comparison CSVs/figures — this is a core Digital Telemetry
     Lead deliverable per the course manual. The stream log is
     filtered to PSK to keep the Monitoring Lead’s feed lean and
     based on the highest-quality modulation only.


FILES PRODUCED
──────────────────────────────────────────────────────────────
  FOR THE MONITORING LEAD
  results/logs/telemetry_stream.log         PSK only — 875 frames

  FOR THE DIGITAL TELEMETRY LEAD (internal / course deliverables)
  results/logs/digital_telemetry_report.json  Full DT chain, all 5 mods
  results/logs/sqnr_comparison.csv            8 / 10 / 12-bit SQNR
  results/logs/line_coding_comparison.csv     NRZ-L vs Manchester BER
  results/figures/sqnr_comparison.png
  results/figures/line_coding_comparison.png
  results/figures/ber_awgn_vs_rayleigh.png
  results/figures/ber_heatmap_stations_x_mods.png
  results/figures/parity_pass_rate_per_mod.png


STREAM LOG FORMAT
──────────────────────────────────────────────────────────────
  JSONL — one JSON object per line.
  Each line = one telemetry frame = one 60-second segment from
  one station, processed through the complete digital telemetry
  chain: Quantisation (8/10/12-bit) → Grey-code PCM →
  NRZ / Manchester line coding → parity check → CRC-16 →
  BER simulation (AWGN + Rayleigh fading).

  Modulation     : PSK only
  Total frames   : 875
  Stations       : EC_1, HBW_1, MM_1, OV_1, SM_1, VGR_1, WT_1
  Simulated span : 2026-03-31T09:00:00  →  2026-03-31T16:17:00
  Frame interval : 30 s


HOW TO STREAM IN PYTHON
──────────────────────────────────────────────────────────────
  # Option A — batch / replay
  import json
  frames = []
  with open("results/logs/telemetry_stream.log") as f:
      for line in f:
          frames.append(json.loads(line.strip()))

  # Option B — simulate live feed
  import json, time
  with open("results/logs/telemetry_stream.log") as f:
      for line in f:
          frame = json.loads(line.strip())
          process(frame)      # your Streamlit update function
          time.sleep(0.5)     # adjust replay speed

  # All frames are PSK — no mod filter needed
  # But you can still filter by station:
  mm1_frames = [f for f in frames if f["station"] == "MM_1"]


FRAME KEYS
──────────────────────────────────────────────────────────────
  log_ts                Original factory timestamp
  recv_ts               Simulated receive timestamp — USE FOR X-AXIS
  station               e.g. "MM_1"
  mod_scheme            Always "PSK" in this stream
  segment_idx           0–124
  segment_start / end   60-second window boundaries
  machine_state         1=READY | 0=FAULT

  DIGITAL TELEMETRY FIELDS
  dt_integrity_flag     1=OK | 0=DEGRADED  (sync-marker check)
  dt_parity_pass_rate   Fraction of 8-bit blocks with even parity [0,1]
  dt_checksum_ok_rate   Fraction of 256-bit frames passing CRC-16  [0,1]
  dt_best_bit_depth     Best quantisation depth (always 12 for PSK)
  dt_sqnr_8bit          SQNR (dB) at 8-bit quantisation
  dt_sqnr_10bit         SQNR (dB) at 10-bit quantisation
  dt_sqnr_12bit         SQNR (dB) at 12-bit quantisation
  dt_channel_ber_awgn   Theoretical BER under AWGN @ 10.0 dB SNR
  dt_channel_ber_fading Theoretical BER under Rayleigh fading @ 10.0 dB
  dt_line_coding_best   "NRZ" or "Manchester" (lower Monte-Carlo BER)
  dt_lc_nrz_ber         NRZ-L Monte-Carlo BER estimate
  dt_lc_manchester_ber  Manchester Monte-Carlo BER estimate

  features              dict — all signal feature columns from the CSV
  reconstructed         dict — PCM-decoded signal stats per signal name
  alerts                list — active alert strings (empty = no alerts)


ALERT LOGIC (Streamlit)
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
    station       multiselect  (EC_1 … WT_1)
    replay_speed  slider  0.1× → 5×
    feature       selectbox  (signal feature to plot)

  Tab 1 — Live Feed
    st.metric   machine_state | dt_integrity_flag | dt_channel_ber_awgn
    st.dataframe  last 20 frames for selected station
    Alert banner  if frame["alerts"]: st.error(...)

  Tab 2 — Feature Trends
    st.line_chart  chosen feature over recv_ts

  Tab 3 — Channel Health
    Bar chart  dt_channel_ber_awgn vs dt_channel_ber_fading per station
    Progress bars  dt_parity_pass_rate | dt_checksum_ok_rate

  Tab 4 — Quantization & PCM
    Bar chart  dt_sqnr_8bit / 10bit / 12bit per station
    Highlight dt_best_bit_depth

  Tab 5 — Line Coding
    Bar chart  dt_lc_nrz_ber vs dt_lc_manchester_ber per station
    Badge  dt_line_coding_best

  Tab 6 — PSD Analysis
    Bar chart  psd_peak and psd_mean_energy per signal

  Tab 7 — Station Heatmap
    st.image("results/figures/ber_heatmap_stations_x_mods.png")


REPLAY SPEED REFERENCE
──────────────────────────────────────────────────────────────
  Real-time   time.sleep(30)    1 frame per 30 s
  Fast        time.sleep(1)     1 frame per second
  Demo        time.sleep(0.2)   5 frames per second
  Instant     time.sleep(0)     static / all-at-once

============================================================
Digital Telemetry Lead · TELE 523 Group 1
