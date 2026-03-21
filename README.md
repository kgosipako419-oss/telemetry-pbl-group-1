# Fischertechnik Smart Factory Dashboard

Place the log file and the Python file in the same folder before doing anything else.

---

## What you need

- `factory_dashboard.py` — the dashboard script
- `low-level_log_20230206-140808.txt` — the factory log data
- Python 3.8 or higher — download from [python.org](https://www.python.org/downloads/) if you don't have it

---

## Install dependencies

Open a terminal, navigate to the folder, and run:

```bash
pip install dash plotly pandas
```

This is a one-time step. Everything else (`json`, `threading`, `datetime`, etc.) is built into Python already.

---

## Run the dashboard

```bash
python factory_dashboard.py --file low-level_log_20230206-140808.txt
```

Then open your browser and go to:

```
http://127.0.0.1:8050


