import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "7612698", "low-level_log_20230206-140808.txt")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
RAW_PLOTS_PATH = os.path.join(BASE_DIR, "data", "raw", "plots")
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(RAW_PLOTS_PATH, exist_ok=True)

# ── Shared plot helper ────────────────────────────────────────────────────────
def plot_all_signals(station_dfs_in, out_dir, title_suffix, fixed_ylim=True, exclude_cols=None):
    """Save one stacked-panel PNG per station to out_dir."""
    _exclude = set(exclude_cols or []) | {"timestamp", "id", "station"}
    colors = cm.tab20.colors

    for station, sdf in station_dfs_in.items():
        signal_cols = [
            c for c in sdf.select_dtypes(include="number").columns
            if c not in _exclude
        ]
        if not signal_cols:
            continue

        n = len(signal_cols)
        row_h = max(1.4, 90 / n)
        fig, axes = plt.subplots(n, 1, figsize=(18, row_h * n), sharex=True)
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Station {station} — {n} signals  [{title_suffix}]",
                     fontsize=13, fontweight="bold", y=1.002)

        x = np.arange(len(sdf))
        for i, (ax, col) in enumerate(zip(axes, signal_cols)):
            color = colors[i % len(colors)]
            vals = pd.to_numeric(sdf[col], errors="coerce").to_numpy()
            ax.plot(x, vals, linewidth=0.8, color=color)
            ax.set_ylabel(col, fontsize=7, rotation=0, labelpad=2, ha="right", va="center")
            if fixed_ylim:
                ax.set_ylim(-0.05, 1.05)
            ax.tick_params(axis="both", labelsize=6)
            ax.grid(axis="x", linewidth=0.3, alpha=0.4)
            # shade failure regions when the raw string label is present
            if "failure_label" in sdf.columns:
                fail_mask = sdf["failure_label"].astype(str).str.strip().ne("") & \
                            sdf["failure_label"].astype(str).str.strip().ne("no_failure")
                ax.fill_between(x, 0, 1, where=fail_mask,
                                color="red", alpha=0.08,
                                transform=ax.get_xaxis_transform())

        axes[-1].set_xlabel("Sample index", fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{station}_signals.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {station} → {out_path}")

def plot_all_signals_overview(station_dfs_in, out_dir, filename, title, exclude_cols=None):
    """One figure, one subplot per station, all signals overlaid on each subplot."""
    _exclude = set(exclude_cols or []) | {"timestamp", "id", "station"}
    stations = list(station_dfs_in.keys())
    n = len(stations)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, station in zip(axes, stations):
        sdf = station_dfs_in[station]
        signal_cols = [
            c for c in sdf.select_dtypes(include="number").columns
            if c not in _exclude
        ]
        colors = cm.tab20(np.linspace(0, 1, max(len(signal_cols), 1)))
        x = np.arange(len(sdf))
        for col, color in zip(signal_cols, colors):
            vals = pd.to_numeric(sdf[col], errors="coerce").to_numpy()
            ax.plot(x, vals, linewidth=0.7, alpha=0.7, label=col, color=color)
        ax.set_title(f"Station: {station}  ({len(signal_cols)} signals)", fontsize=10)
        ax.set_xlabel("Sample index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(loc="upper right", fontsize=6, ncol=3, framealpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview chart → {out_path}")

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

# snapshot before any transforms so we can plot raw signals
raw_station_dfs = {k: v.copy() for k, v in station_dfs.items()}
print("  Plotting raw signals…")
plot_all_signals(
    raw_station_dfs,
    out_dir=RAW_PLOTS_PATH,
    title_suffix="RAW — original units",
    fixed_ylim=False,
    exclude_cols=["current_state"],
)
print(f"  Raw signal charts saved to data/raw/plots/")
plot_all_signals_overview(
    raw_station_dfs,
    out_dir=RAW_PLOTS_PATH,
    filename="all_signals_raw.png",
    title="All Stations — All Signals (RAW, before preprocessing)",
    exclude_cols=["current_state"],
)
print("  Raw overview chart saved to data/raw/plots/all_signals_raw.png")

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

# ── Step 9 · Plot processed signals ──────────────────────────────────────────
print("  Plotting processed signals…")
plot_all_signals(
    station_dfs,
    out_dir=PROCESSED_PATH,
    title_suffix="PROCESSED — normalised 0–1",
    fixed_ylim=True,
    exclude_cols=BOOL_COLS + STR_COLS + ["current_state_binary"],
)
print("Step 9 done — processed signal charts saved to data/processed/")