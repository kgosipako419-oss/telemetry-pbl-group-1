import json
import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "7612698", "low-level_log_20230206-140808.txt")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ── Step 1 · Load & Parse ─────────────────────────────────────────────────────
records = []
with open(RAW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records.append(record)

df = pd.DataFrame(records)
print(f"Step 1 done — {len(df)} records loaded")

# ── Step 2 · Separate by station ──────────────────────────────────────────────
station_dfs = {station: grp.reset_index(drop=True)
               for station, grp in df.groupby("station")}
print(f"Step 2 done — {len(station_dfs)} stations: {list(station_dfs.keys())}")

# ── Step 3 · Parse timestamps ─────────────────────────────────────────────────
for station, sdf in station_dfs.items():
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    station_dfs[station] = sdf.sort_values("timestamp").reset_index(drop=True)
print("Step 3 done — timestamps parsed and sorted")

# ── Step 4 · Handle missing values ────────────────────────────────────────────
for station, sdf in station_dfs.items():
    num_cols = sdf.select_dtypes(include="number").columns
    sdf[num_cols] = sdf[num_cols].ffill().bfill()
    obj_cols = sdf.select_dtypes(include="object").columns
    sdf[obj_cols] = sdf[obj_cols].infer_objects(copy=False).fillna("")
    station_dfs[station] = sdf
print("Step 4 done — missing values handled")

# ── Step 5 · Encode categoricals ──────────────────────────────────────────────
BOOL_COLS = [
    "i1_light_barrier", "i1_pos", "i1_pos_switch",
    "i2_light_barrier", "i2_pos", "i2_pos_switch",
    "i3_light_barrier", "i3_pos_switch",
    "i4_light_barrier", "i4_pos_switch",
    "i5_joystick_x_f", "i5_light_barrier", "i5_pos_switch",
    "i6_joystick_y_f", "i6_light_barrier", "i6_pos_switch",
    "i7_joystick_x_b", "i7_light_barrier", "i7_pos_switch",
    "i8_joystick_y_b", "i8_light_barrier", "i8_pos_switch",
    "o5_valve", "o6_valve", "o7_valve", "o8_valve_open",
]
STR_COLS = ["current_task", "current_sub_task", "failure_label"]

for station, sdf in station_dfs.items():
    for col in BOOL_COLS:
        if col in sdf.columns:
            # replace empty strings with NaN, then cast bool → 0/1, keep NaN as pd.NA
            sdf[col] = sdf[col].replace("", pd.NA)
            sdf[col] = sdf[col].map(lambda x: 1 if x is True else (0 if x is False else pd.NA))
            sdf[col] = sdf[col].astype("Int8")
    for col in STR_COLS:
        if col in sdf.columns:
            sdf[col] = pd.Categorical(sdf[col]).codes
    station_dfs[station] = sdf
print("Step 5 done — categoricals encoded")

# ── Step 6 · Normalise signals ────────────────────────────────────────────────
EXCLUDE_FROM_NORM = BOOL_COLS + STR_COLS + ["current_state_binary"]

for station, sdf in station_dfs.items():
    num_cols = sdf.select_dtypes(include="number").columns
    cols_to_scale = [c for c in num_cols if c not in EXCLUDE_FROM_NORM]
    for col in cols_to_scale:
        col_min = sdf[col].min()
        col_max = sdf[col].max()
        if col_max - col_min > 0:
            sdf[col] = (sdf[col] - col_min) / (col_max - col_min)
    station_dfs[station] = sdf
print("Step 6 done — signals normalised (min-max)")

# ── Step 7 · Label target variable ────────────────────────────────────────────
for station, sdf in station_dfs.items():
    sdf["current_state_binary"] = (sdf["current_state"] == "ready").astype(int)
    sdf = sdf.drop(columns=["current_state"])
    station_dfs[station] = sdf
print("Step 7 done — target variable labelled (ready=1, not ready=0)")

# ── Step 8 · Save to data/processed/ ─────────────────────────────────────────
for station, sdf in station_dfs.items():
    out_path = os.path.join(PROCESSED_PATH, f"{station}.csv")
    sdf.to_csv(out_path, index=False)
    print(f"  Saved {station} → {out_path}  ({len(sdf)} rows, {len(sdf.columns)} cols)")

print("\nPhase 1 complete — all stations saved to data/processed/")