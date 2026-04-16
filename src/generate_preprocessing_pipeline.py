"""
Image Generation — Preprocessing & Signal Processing Pipeline Visuals
======================================================================
Produces four figures in results/:
  1. PREPROCESSING_pipeline.png              — full step-by-step pipeline diagram
  2. PREPROCESSING_results_summary.png       — bar charts + step tables
  3. PREPROCESSING_interconnection_web.png   — station data-flow web
  4. SIGNAL_PROCESSING_filtered_signals.png  — all signals after FIR filtering
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "results")
os.makedirs(OUT, exist_ok=True)

# ── Shared config (must match phase2_signal_processing.py exactly) ─────────
FS           = 0.5
NYQUIST      = FS / 2
CUTOFF       = 0.1
FIR_NUMTAPS  = 51
WINDOW_SIZE  = 30
OVERLAP      = 15
STEP         = WINDOW_SIZE - OVERLAP
WELCH_NPERSEG = 16

STATIONS  = ["EC_1", "HBW_1", "MM_1", "OV_1", "SM_1", "VGR_1", "WT_1"]
ROWS      = [1886, 1881, 1886, 1885, 1885, 1882, 1885]
SIG_COLS  = [2, 9, 5, 3, 3, 10, 3]
# Features per segment = 7 per signal col (rms, mean, var, fft_peak,
#                                           psd_peak, psd_mean_energy, psd_peak_freq)
FEATURES  = [c * 7 for c in SIG_COLS]   # [14, 63, 35, 21, 21, 70, 21]
# Approximate segment count per station (floor((rows - window) / step) + 1)
SEGMENTS  = [max(0, (r - WINDOW_SIZE) // STEP + 1) for r in ROWS]
COLORS    = ["#4a5fc1","#3a8fd1","#2ebd7a","#e8a838","#e85d3a","#9b5de5","#f72585"]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def draw_box(ax, x, y, w, h, label, sublabel, color,
             text_color="#ffffff", radius=0.3):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=1.5, edgecolor=color,
        facecolor=color + "33", zorder=3)
    ax.add_patch(box)
    ax.text(x, y + 0.12, label,    ha="center", va="center",
            fontsize=11, fontweight="bold", color=text_color, zorder=4)
    ax.text(x, y - 0.25, sublabel, ha="center", va="center",
            fontsize=8.5, color="#aaaaaa", zorder=4)


def draw_arrow(ax, x, y_start, y_end, color="#555577"):
    ax.annotate("", xy=(x, y_end + 0.02), xytext=(x, y_start - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
                zorder=2)


def style_table(table, header_color, row_colors):
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(header_color if r == 0
                           else row_colors[r % 2])
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333333")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 1 — Pipeline Diagram
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 24))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xlim(0, 10)
ax.set_ylim(0, 26)
ax.axis("off")

ax.text(5, 25.3, "Signal Processing Lead · Full Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#ffffff")
ax.text(5, 24.85, "TELE 523 · Group 5 · Industrial Machine Condition Monitoring",
        ha="center", va="center", fontsize=10, color="#888888")

draw_box(ax, 5, 24.2, 6, 0.7, "Input · raw dataset",
         "data/raw/7612698/low-level_log_20230206-140808.txt",
         "#444455", "#ccccff")

# ── Phase 1 background ───────────────────────────────────────────────────────
phase1_bg = FancyBboxPatch((1.2, 13.3), 7.6, 10.4,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    linewidth=1, edgecolor="#3a3a6a", facecolor="#1a1a3a", zorder=1)
ax.add_patch(phase1_bg)
ax.text(2.2, 23.45, "Phase 1 — Data Preprocessing · src/preprocessing.py",
        fontsize=8.5, color="#8888cc", zorder=4)

PHASE1_COLOR = "#4a5fc1"
steps_p1 = [
    (22.9, "Step 1 · Load & Parse",         "JSON lines → Pandas DataFrame"),
    (21.8, "Step 2 · Separate by station",   "Split into 7 per-machine DataFrames"),
    (20.7, "Step 3 · Parse timestamps",      "Datetime conversion, sort by time"),
    (19.6, "Step 4 · Handle missing values", "Fill or drop NaNs per station"),
    (18.5, "Step 5 · Encode categoricals",   "Booleans → 0/1, strings → integers"),
    (17.4, "Step 6 · Normalise signals",     "Min-max scaling on sensor columns"),
    (16.3, "Step 7 · Label target variable", "current_state → binary flag (ready=1)"),
]

draw_arrow(ax, 5, 23.85, 23.25)
for i, (y, lbl, sub) in enumerate(steps_p1):
    draw_box(ax, 5, y, 6.5, 0.75, lbl, sub, PHASE1_COLOR)
    if i < len(steps_p1) - 1:
        draw_arrow(ax, 5, y - 0.38, steps_p1[i+1][0] + 0.38)

draw_arrow(ax, 5, 15.92, 15.28)
draw_box(ax, 5, 14.95, 6.5, 0.75,
         "Step 8 · Save to data/processed/",
         "Per-station clean CSVs exported (7 files)",
         "#2e7d32", "#aaffaa")

# ── Phase 2 background ───────────────────────────────────────────────────────
phase2_bg = FancyBboxPatch((1.2, 3.2), 7.6, 10.3,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    linewidth=1, edgecolor="#1a5a3a", facecolor="#0d2a1a", zorder=1)
ax.add_patch(phase2_bg)
ax.text(2.2, 13.2, "Phase 2 — Signal Processing · src/signal_processing.py",
        fontsize=8.5, color="#55aa77", zorder=4)

PHASE2_COLOR = "#1b7a4a"
steps_p2 = [
    (12.6, "Step 9 · Sampling",
           f"Resample to 2 s grid | fs={FS} Hz | Nyquist={NYQUIST} Hz | cutoff={CUTOFF} Hz"),
    (11.3, "Step 10 · FIR Low-Pass Filter",
           f"firwin (Hamming window) | numtaps={FIR_NUMTAPS} | cutoff={CUTOFF} Hz | linear phase"),
    (10.0, "Step 11 · Segmentation",
           f"Sliding window | size={WINDOW_SIZE} samples ({WINDOW_SIZE*2} s) | "
           f"overlap={OVERLAP} | step={STEP}"),
    (8.7,  "Step 12 · Welch PSD",
           f"scipy.signal.welch | nperseg={WELCH_NPERSEG} | "
           f"features: psd_peak, psd_mean_energy, psd_peak_freq"),
    (7.4,  "Step 13 · Feature Extraction",
           "Per segment: RMS, mean, variance, FFT peak  +  3 PSD features = 7 per signal"),
]

draw_arrow(ax, 5, 14.57, 13.05)
for i, (y, lbl, sub) in enumerate(steps_p2):
    draw_box(ax, 5, y, 6.5, 0.85, lbl, sub, PHASE2_COLOR, "#aaffcc")
    if i < len(steps_p2) - 1:
        draw_arrow(ax, 5, y - 0.43, steps_p2[i+1][0] + 0.43, "#335544")

draw_arrow(ax, 5, 6.97, 6.35)
draw_box(ax, 5, 5.95, 6.5, 0.75,
         "Output · conditioned signal",
         "7 × *_filtered.csv (FIR-filtered)  +  7 × *_features.csv (segmented)",
         "#444455", "#ccccff")

draw_arrow(ax, 5, 5.57, 4.95)
draw_box(ax, 5, 4.6, 6.5, 0.65,
         "Handoff to Modulation Lead",
         "Feature vectors ready for BPSK / QPSK / 16-QAM modulation pipeline",
         "#5a3a1a", "#ffddaa")

plt.tight_layout(pad=0)
plt.savefig(os.path.join(OUT, "PREPROCESSING_pipeline.png"),
            dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Image 1 saved → results/PREPROCESSING_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 2 — Results Summary
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(18, 13))
fig2.patch.set_facecolor("#0d1117")
fig2.suptitle(
    "Preprocessing & Signal Processing — Results Summary\nTELE 523 · Group 5",
    fontsize=16, fontweight="bold", color="white", y=0.98)

# ── Plot 1 · Records per station ─────────────────────────────────────────────
ax1 = axes[0, 0]
ax1.set_facecolor("#161b22")
bars = ax1.bar(STATIONS, ROWS, color=COLORS, edgecolor="#ffffff22", linewidth=0.8)
ax1.set_title("Records per Station (after Phase 1 preprocessing)",
              color="white", fontsize=11, pad=10)
ax1.set_ylabel("Row count", color="#aaaaaa")
ax1.tick_params(colors="#aaaaaa")
ax1.spines[:].set_color("#333333")
ax1.set_ylim(1870, 1895)
for bar, val in zip(bars, ROWS):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha="center", va="bottom", color="white", fontsize=8.5)

# ── Plot 2 · Segments + features per station ─────────────────────────────────
ax2 = axes[0, 1]
ax2.set_facecolor("#161b22")
x  = np.arange(len(STATIONS))
w  = 0.28
b1 = ax2.bar(x - w, SIG_COLS,  w, label="Signal cols filtered", color="#4a5fc1",
             edgecolor="#ffffff22")
b2 = ax2.bar(x,     SEGMENTS,  w, label=f"Segments (win={WINDOW_SIZE}, step={STEP})",
             color="#e8a838", edgecolor="#ffffff22")
b3 = ax2.bar(x + w, FEATURES,  w, label="Feature cols per segment", color="#2ebd7a",
             edgecolor="#ffffff22")
ax2.set_title("Signal Cols · Segments · Features per Station",
              color="white", fontsize=11, pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(STATIONS)
ax2.tick_params(colors="#aaaaaa")
ax2.spines[:].set_color("#333333")
ax2.legend(facecolor="#222233", labelcolor="white", fontsize=8.5)
for bar in list(b1) + list(b2) + list(b3):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             str(int(bar.get_height())),
             ha="center", va="bottom", color="white", fontsize=7)

# ── Table 1 · Phase 1 steps ───────────────────────────────────────────────────
ax3 = axes[1, 0]
ax3.set_facecolor("#161b22")
ax3.axis("off")
ax3.set_title("Phase 1 — Preprocessing Steps", color="white", fontsize=11, pad=10)
p1_steps = [
    ["Step 1", "Load & Parse",          "JSON lines → DataFrame (~13 k records)"],
    ["Step 2", "Separate by station",   "7 DataFrames: EC_1 … WT_1"],
    ["Step 3", "Parse timestamps",      "pd.to_datetime + sort_values"],
    ["Step 4", "Handle missing values", "ffill/bfill numeric, fillna('') strings"],
    ["Step 5", "Encode categoricals",   "26 bool cols → Int8, 3 str cols → codes"],
    ["Step 6", "Normalise signals",     "Min-max scaling, range [0, 1]"],
    ["Step 7", "Label target",          "current_state → binary (ready=1)"],
    ["Step 8", "Save CSVs",             "7 × 51-col CSVs → data/processed/"],
]
t1 = ax3.table(cellText=p1_steps, colLabels=["#", "Step", "What it does"],
               loc="center", cellLoc="left")
style_table(t1, header_color="#2a2a5a",
            row_colors=["#1e2530", "#161b22"])

# ── Table 2 · Phase 2 steps ───────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.set_facecolor("#161b22")
ax4.axis("off")
ax4.set_title("Phase 2 — Signal Processing Steps", color="white", fontsize=11, pad=10)
p2_steps = [
    ["Step 9",  "Sampling",
     f"resample('2s'), fs={FS} Hz, Nyquist={NYQUIST} Hz"],
    ["Step 10", "FIR Low-Pass Filter",
     f"firwin Hamming, numtaps={FIR_NUMTAPS}, cutoff={CUTOFF} Hz"],
    ["Step 11", "Segmentation",
     f"Sliding window: size={WINDOW_SIZE} samp, overlap={OVERLAP}, step={STEP}"],
    ["Step 12", "Welch PSD",
     f"scipy.signal.welch, nperseg={WELCH_NPERSEG} — psd_peak, mean_energy, peak_freq"],
    ["Step 13", "Feature Extraction",
     "RMS, mean, variance, FFT peak + 3 PSD = 7 features/col/segment"],
    ["Output",  "Filtered signals",
     "7 × *_filtered.csv (FIR-filtered full signal)"],
    ["Output",  "Feature vectors",
     f"7 × *_features.csv (~{SEGMENTS[0]} rows each, one row per segment)"],
]
t2 = ax4.table(cellText=p2_steps, colLabels=["#", "Step", "What it does"],
               loc="center", cellLoc="left")
style_table(t2, header_color="#1a4a2a",
            row_colors=["#1e2e20", "#161b22"])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "PREPROCESSING_results_summary.png"),
            dpi=180, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print("Image 2 saved → results/PREPROCESSING_results_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 3 — Station Interconnection Web
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3w = plt.subplots(figsize=(16, 14))
fig3.patch.set_facecolor("#0d1117")
ax3w.set_facecolor("#0d1117")
ax3w.set_xlim(-1.6, 1.6)
ax3w.set_ylim(-1.6, 1.6)
ax3w.axis("off")

ax3w.text(0, 1.52, "Station Interconnection Web",
          ha="center", va="center", fontsize=18, fontweight="bold", color="white")
ax3w.text(0, 1.42,
          "TELE 523 · Group 5 · Data flow between machines in the production line",
          ha="center", va="center", fontsize=10, color="#888888")

STATIONS_WEB   = ["MM_1", "OV_1", "SM_1", "VGR_1", "HBW_1", "WT_1", "EC_1"]
STATION_COLORS = {
    "MM_1":  "#4a5fc1", "OV_1":  "#3a8fd1", "SM_1":  "#2ebd7a",
    "VGR_1": "#e8a838", "HBW_1": "#e85d3a", "WT_1":  "#9b5de5",
    "EC_1":  "#f72585",
}
STATION_ROLES = {
    "MM_1":  "Milling\nMachine",  "OV_1":  "Oven",
    "SM_1":  "Sorting\nMachine",  "VGR_1": "Vacuum\nGripper",
    "HBW_1": "High-Bay\nWarehouse","WT_1": "Work\nTransfer",
    "EC_1":  "Environment\nController",
}

n      = len(STATIONS_WEB)
angles = [2 * np.pi * i / n - np.pi/2 for i in range(n)]
r      = 1.1
positions = {s: (r * np.cos(a), r * np.sin(a))
             for s, a in zip(STATIONS_WEB, angles)}

ax3w.add_patch(plt.Circle((0, 0), 0.18, color="#1a1a3a", zorder=3,
                           linewidth=2, ec="#6666cc"))
ax3w.text(0,  0.04, "RAW",  ha="center", va="center", fontsize=10,
          fontweight="bold", color="#aaaaff", zorder=4)
ax3w.text(0, -0.07, "LOG",  ha="center", va="center", fontsize=10,
          fontweight="bold", color="#aaaaff", zorder=4)

FLOW = [
    ("HBW_1", "VGR_1", "workpiece\npickup",  "#e85d3a"),
    ("VGR_1", "MM_1",  "to milling",          "#e8a838"),
    ("MM_1",  "OV_1",  "to oven",             "#4a5fc1"),
    ("OV_1",  "SM_1",  "to sorting",          "#3a8fd1"),
    ("SM_1",  "HBW_1", "store back",          "#2ebd7a"),
    ("VGR_1", "WT_1",  "transfer",            "#e8a838"),
    ("WT_1",  "SM_1",  "sorted piece",        "#9b5de5"),
    ("EC_1",  "MM_1",  "env. control",        "#f72585"),
    ("EC_1",  "OV_1",  "env. control",        "#f72585"),
]

for src, dst, label, color in FLOW:
    x0, y0 = positions[src]
    x1, y1 = positions[dst]
    ax3w.annotate("",
        xy=(x1 * 0.82, y1 * 0.82), xytext=(x0 * 0.82, y0 * 0.82),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                        connectionstyle="arc3,rad=0.25"), zorder=2)
    mx = (x0 + x1) / 2 * 0.72
    my = (y0 + y1) / 2 * 0.72
    ax3w.text(mx, my, label, ha="center", va="center", fontsize=6.5,
              color=color,
              bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117",
                        edgecolor=color+"55", linewidth=0.8), zorder=5)

# Node bubbles — now show segments count instead of old feature count
FEAT_MAP = dict(zip(STATIONS, FEATURES))
SEG_MAP  = dict(zip(STATIONS, SEGMENTS))

NODE_R = 0.155
for s, (x, y) in positions.items():
    color = STATION_COLORS[s]
    ax3w.add_patch(plt.Circle((x, y), NODE_R, color=color+"33",
                               zorder=3, linewidth=2.5, ec=color))
    ax3w.text(x, y + 0.04, s,
              ha="center", va="center", fontsize=9,
              fontweight="bold", color="white", zorder=4)
    ax3w.text(x, y - 0.06, STATION_ROLES[s],
              ha="center", va="center", fontsize=6.5,
              color="#aaaaaa", zorder=4)
    ox  = x * 1.38
    oy  = y * 1.38
    seg = SEG_MAP.get(s, "?")
    ft  = FEAT_MAP.get(s, "?")
    info = f"{seg} segments\n{ft} features"
    ax3w.text(ox, oy, info, ha="center", va="center", fontsize=7,
              color=color,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#161b22",
                        edgecolor=color, linewidth=1), zorder=5)

legend_items = [
    mpatches.Patch(color=STATION_COLORS[s],
                   label=f"{s} — {STATION_ROLES[s].replace(chr(10), ' ')}")
    for s in STATIONS_WEB
]
ax3w.legend(handles=legend_items, loc="lower center", ncol=4,
            facecolor="#161b22", labelcolor="white", fontsize=8,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout(pad=1)
plt.savefig(os.path.join(OUT, "PREPROCESSING_interconnection_web.png"),
            dpi=180, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()
print("Image 3 saved → results/PREPROCESSING_interconnection_web.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 4 — All Signals After Signal Processing (FIR-filtered)
# ══════════════════════════════════════════════════════════════════════════════
SIGNAL_COLS = {
    "EC_1"  : ["i3_photoresistor", "current_task_duration"],
    "HBW_1" : ["m1_speed", "m2_speed", "m3_speed", "m4_speed",
                "current_pos_x", "current_pos_y",
                "target_pos_x", "target_pos_y", "current_task_duration"],
    "MM_1"  : ["m1_speed", "m2_speed", "m3_speed",
                "o8_compressor", "current_task_duration"],
    "OV_1"  : ["m1_speed", "o8_compressor", "current_task_duration"],
    "SM_1"  : ["m1_speed", "o8_compressor", "current_task_duration"],
    "VGR_1" : ["m1_speed", "m2_speed", "m3_speed",
                "current_pos_x", "current_pos_y", "current_pos_z",
                "target_pos_x", "target_pos_y", "target_pos_z",
                "current_task_duration"],
    "WT_1"  : ["m2_speed", "o8_compressor", "current_task_duration"],
}

PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

TAB20 = plt.get_cmap("tab20")

n_stations = len(STATIONS)
# one column per station, rows = max signal count across all stations
max_sigs = max(len(v) for v in SIGNAL_COLS.values())
n_cols = 2
n_rows = (n_stations + 1) // n_cols   # ceil divide

fig4, axes4 = plt.subplots(
    n_rows, n_cols,
    figsize=(22, 5 * n_rows),
)
fig4.patch.set_facecolor("#0d1117")
fig4.suptitle(
    "All Signals After Signal Processing  (FIR low-pass filtered, resampled to 2 s grid)\n"
    "TELE 523 · Group 5",
    fontsize=14, fontweight="bold", color="white", y=1.01,
)

axes4_flat = axes4.flatten()

for idx, station in enumerate(STATIONS):
    ax = axes4_flat[idx]
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#333333")
    ax.tick_params(colors="#aaaaaa", labelsize=7)

    filtered_csv = os.path.join(PROCESSED_PATH, f"{station}_filtered.csv")
    if not os.path.exists(filtered_csv):
        ax.text(0.5, 0.5, f"{station}\n(filtered CSV not found)",
                ha="center", va="center", color="#ff6666",
                transform=ax.transAxes, fontsize=10)
        ax.set_title(station, color="white", fontsize=11)
        continue

    sdf = pd.read_csv(filtered_csv)
    sig_cols = [c for c in SIGNAL_COLS[station] if c in sdf.columns]
    x = np.arange(len(sdf))

    for i, col in enumerate(sig_cols):
        color = TAB20(i / max(len(sig_cols), 1))
        vals = pd.to_numeric(sdf[col], errors="coerce").to_numpy()
        ax.plot(x, vals, linewidth=0.9, alpha=0.85, label=col, color=color)

    ax.set_title(f"{station}  ({len(sig_cols)} signals)", color="white",
                 fontsize=11, pad=6)
    ax.set_xlabel("Sample index", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Filtered value", color="#aaaaaa", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=2,
              facecolor="#1e2530", labelcolor="white", framealpha=0.7)

# hide any unused subplot cells
for idx in range(n_stations, len(axes4_flat)):
    axes4_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "SIGNAL_PROCESSING_filtered_signals.png"),
            dpi=180, bbox_inches="tight", facecolor=fig4.get_facecolor())
plt.close()
print("Image 4 saved → results/SIGNAL_PROCESSING_filtered_signals.png")