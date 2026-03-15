```
TELE 523 Telemetry and Remote Control
Python-Based PBL Laboratory — Group 1
Botswana International University of Science and Technology
---

## PROJECT TITLE
Industrial Machine Condition Monitoring Using Telemetry for Mining and Power Generation in Botswana

---
 OVERVIEW

This repository contains the semester-long Problem Based Learning laboratory project for TELE 523.
The project designs and simulates a complete end-to-end telemetry and remote monitoring system
using Python, applied to the domain of industrial machine condition monitoring.

The system ingests real sensor data from the Fischertechnik Smart Factory IoT dataset, processes
it through analog and digital modulation and demodulation stages, applies digital telemetry
techniques including quantization and PCM encoding, extracts machine condition features, and
displays results on a Streamlit monitoring dashboard with threshold-based alerting.

**Pipeline:**
Dataset → Preprocessing → Modulation → Channel → Demodulation → Digital Telemetry → Feature Extraction → Monitoring Output

---

Group Members

| Name | Role |
|---|---|
| Pako Kgosintwa | System Architect |
| Thebe Ratsatsi | Signal Processing Lead |
| Goitse Pihelo | Modulation Lead |
| Atlang Zambezi | Digital Telemetry Lead |
| Tsotlhe Seiphepi | Monitoring Lead |

---
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── figures/
│   └── logs/
│
├── scripts/
│
├── src/
│   ├── signal_processing_lead/
│   │   ├── preprocessing.py
│   │   ├── gap_analysis.py
│   │   ├── prepare_psd_ready.py
│   │   ├── filter_compare.py
│   │   └── filter_metrics.py
│   │
│   ├── modulation_lead/
│   │   ├── am_modulation.py
│   │   ├── fm_modulation.py
│   │   ├── ask_modulation.py
│   │   ├── fsk_modulation.py
│   │   ├── psk_modulation.py
│   │   └── channel_noise.py
│   │
│   ├── digital_telemetry_lead/
│   │   ├── quantization.py
│   │   ├── pcm_encoding.py
│   │   ├── line_coding.py
│   │   └── bit_integrity_check.py
│   │
│   ├── monitoring_lead/
│   │   ├── feature_extraction.py
│   │   ├── threshold_detection.py
│   │   ├── drift_detection.py
│   │   ├── alert_system.py
│   │   └── dashboard_streamlit.py
│   │
│   └── system_architect/
│       ├── pipeline_controller.py
│       ├── integration_tests.py
│       └── system_diagram.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_modulation.py
│   ├── test_telemetry_pipeline.py
│   └── test_monitoring.py
│
├── requirements.txt
└── README.md
```
