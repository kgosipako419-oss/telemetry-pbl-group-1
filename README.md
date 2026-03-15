```
telemetry_pbl_group1/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── dataset_description.md
│   ├── report_figures.md
│   └── system_architecture.md
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
