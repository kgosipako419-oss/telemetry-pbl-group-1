# TELE 523 Telemetry PBL – Group 5

## Project Title
Industrial Machine Condition Monitoring Using Telemetry for Mining and Power Generation in Botswana

## Overview
This repository contains the end-to-end telemetry and remote monitoring pipeline implemented in Python:
Dataset → Preprocessing → Modulation → Channel → Demodulation → Digital Telemetry (Quantization/PCM) → Feature Extraction → Monitoring Output.

## Repository Structure
'''
telemetry_pbl_group1/
│
├── data/
│   ├── raw/                  # Original dataset files
│   └── processed/            # Final processed datasets used by downstream modules
│
├── docs/                     # Notes, design documents, report assets
│   ├── dataset_description.md
│   ├── report_figures.md
│   └── system_architecture.md
├── results/
│   ├── figures/              # Generated plots and diagrams                
│   └── logs/                 # Execution logs
│
├── scripts/                  # Top-level runnable scripts for pipeline execution
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
├── tests/                    # Additional test scripts and validation cases
│   ├── test_preprocessing.py
│   ├── test_modulation.py
│   ├── test_telemetry_pipeline.py
│   └── test_monitoring.py  
'''
├── requirements.txt          # Python dependencies
└── README.md
