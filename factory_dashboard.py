"""
Fischertechnik Smart Factory — Live Playback Dashboard
=======================================================
Simulates real-time data streaming by reading through the log file
line by line, as if the factory is running live.

REQUIREMENTS:
    pip install dash plotly pandas

RUN:
    python factory_dashboard.py --file low-level_log_20230206-140808.txt

Then open:  http://127.0.0.1:8050
"""

import json
import argparse
import threading
import time
import collections
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash

# ── CONFIG ──────────────────────────────────────────────────────────────────

STATIONS = ["MM_1", "HBW_1", "VGR_1", "SM_1", "OV_1", "WT_1", "EC_1"]

STATION_LABELS = {
    "MM_1":  "MM_1 · Milling Machine",
    "HBW_1": "HBW_1 · Warehouse",
    "VGR_1": "VGR_1 · Robot Arm",
    "SM_1":  "SM_1 · Sorting Machine",
    "OV_1":  "OV_1 · Oven",
    "WT_1":  "WT_1 · Transfer Unit",
    "EC_1":  "EC_1 · Env. Controller",
}

STATION_COLORS = {
    "MM_1":  "#6366f1",
    "HBW_1": "#0ea5e9",
    "VGR_1": "#10b981",
    "SM_1":  "#f59e0b",
    "OV_1":  "#ef4444",
    "WT_1":  "#8b5cf6",
    "EC_1":  "#64748b",
}

# How many historical points to keep in rolling charts
MAX_HISTORY = 300

# ── SHARED STATE ─────────────────────────────────────────────────────────────

# Latest reading per station
latest = {s: {} for s in STATIONS}

# Rolling history: station -> deque of (timestamp, field, value)
history = {s: collections.deque(maxlen=MAX_HISTORY) for s in STATIONS}

# Motor speed history: station -> {motor: deque of (ts, speed)}
motor_history = {s: collections.defaultdict(lambda: collections.deque(maxlen=MAX_HISTORY))
                 for s in STATIONS}

# Position history: VGR_1 and HBW_1
pos_history = {
    "VGR_1": {"x": collections.deque(maxlen=MAX_HISTORY),
               "y": collections.deque(maxlen=MAX_HISTORY),
               "z": collections.deque(maxlen=MAX_HISTORY),
               "ts": collections.deque(maxlen=MAX_HISTORY)},
    "HBW_1": {"x": collections.deque(maxlen=MAX_HISTORY),
               "y": collections.deque(maxlen=MAX_HISTORY),
               "ts": collections.deque(maxlen=MAX_HISTORY)},
}

# Failure log: list of {ts, station, label}
failure_log = collections.deque(maxlen=200)

# Gantt timeline: accumulated sub-task blocks {station, task, sub_task, start, end}
gantt_tasks = collections.deque(maxlen=600)
# Track the currently active sub-task per station to detect transitions
_active_sub = {s: {"task": None, "sub": None, "start": None} for s in STATIONS}

# ── STATE MACHINE ────────────────────────────────────────────────────────────

# Canonical sub-task step sequences per station (derived from log analysis)
STATION_STEPS = {
    "MM_1":  ["idle", "transport to mill", "milling", "eject", "transport to sorter", "return home"],
    "HBW_1": ["idle", "move to slot", "pick up bucket", "transport to belt", "drop off", "wait for VGR", "transport to crane jib", "pick up from belt", "transport to slot", "drop off at slot"],
    "VGR_1": ["idle", "calibrate", "move to pickup", "pick up workpiece", "transport", "drop off"],
    "SM_1":  ["idle", "detect color", "transport to sink", "eject to sink"],
    "OV_1":  ["idle", "move to oven", "open door", "load workpiece", "close door", "burning", "open door", "unload", "close door"],
    "WT_1":  ["idle", "conveyor moving", "valve active", "transfer done"],
    "EC_1":  ["idle", "monitoring"],
}

# Maps sub-task keywords -> canonical step name per station
STEP_KEYWORDS = {
    "MM_1": [
        ("transport.*mill",       "transport to mill"),
        ("milling",               "milling"),
        ("eject",                 "eject"),
        ("transport.*sort",       "transport to sorter"),
        ("ejection",              "eject"),
        ("initial",               "return home"),
    ],
    "HBW_1": [
        ("moving towards",        "move to slot"),
        ("picking up.*slot",      "pick up bucket"),
        ("transport.*conveyor",   "transport to belt"),
        ("dropping.*conveyor",    "drop off"),
        ("waiting",               "wait for VGR"),
        ("transport.*crane",      "transport to crane jib"),
        ("picking up.*conveyor",  "pick up from belt"),
        ("transport.*slot",       "transport to slot"),
        ("dropping.*slot",        "drop off at slot"),
    ],
    "VGR_1": [
        ("calibrat",              "calibrate"),
        ("moving towards",        "move to pickup"),
        ("picking up",            "pick up workpiece"),
        ("transport",             "transport"),
        ("dropping off",          "drop off"),
    ],
    "SM_1": [
        ("detect",                "detect color"),
        ("transport.*sink",       "transport to sink"),
        ("eject",                 "eject to sink"),
    ],
    "OV_1": [
        ("moving towards",        "move to oven"),
        ("open.*door",            "open door"),
        ("transport.*inside",     "load workpiece"),
        ("close.*door|closing",   "close door"),
        ("burning",               "burning"),
        ("transport.*outside",    "unload"),
        ("picking up",            "pick up"),
        ("transport.*milling",    "transport to milling"),
        ("dropping off",          "drop off"),
    ],
    "WT_1": [
        ("conveyor|motor",        "conveyor moving"),
        ("valve",                 "valve active"),
    ],
    "EC_1": [
        ("",                      "monitoring"),
    ],
}

# Live state machine state per station
sm_state = {s: {"step_label": "idle", "task": "", "sub": "", "elapsed": 0.0,
                 "state": "ready", "history": []} for s in STATIONS}


# Playback control
playback = {
    "running": False,
    "speed":   5.0,       # multiplier vs real time
    "current_line": 0,
    "total_lines":  0,
    "log_file":     None,
    "paused":       False,
}

playback_lock = threading.Lock()


# ── PLAYBACK ENGINE ──────────────────────────────────────────────────────────

def run_playback(filepath: str):
    """Background thread: reads log file line by line and populates shared state."""
    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    playback["total_lines"] = len(lines)

    prev_ts = None
    for idx, line in enumerate(lines):
        # Check pause / stop
        while playback["paused"] and playback["running"]:
            time.sleep(0.1)
        if not playback["running"]:
            break

        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        station = obj.get("station")
        if station not in STATIONS:
            continue

        ts_str = obj.get("timestamp", "")
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            ts = datetime.now()

        # Throttle to simulate real-time at chosen speed
        if prev_ts is not None:
            delta = (ts - prev_ts).total_seconds()
            if delta > 0:
                sleep_time = delta / playback["speed"]
                time.sleep(max(0, sleep_time))
        prev_ts = ts

        with playback_lock:
            latest[station] = obj
            playback["current_line"] = idx + 1

            # Motor speeds
            for key in ("m1_speed", "m2_speed", "m3_speed", "m4_speed"):
                if key in obj:
                    motor_history[station][key].append((ts, obj[key]))

            # Position
            if station == "VGR_1":
                pos_history["VGR_1"]["x"].append(obj.get("current_pos_x", 0))
                pos_history["VGR_1"]["y"].append(obj.get("current_pos_y", 0))
                pos_history["VGR_1"]["z"].append(obj.get("current_pos_z", 0))
                pos_history["VGR_1"]["ts"].append(ts)
            if station == "HBW_1":
                pos_history["HBW_1"]["x"].append(obj.get("current_pos_x", 0))
                pos_history["HBW_1"]["y"].append(obj.get("current_pos_y", 0))
                pos_history["HBW_1"]["ts"].append(ts)

            # Failures
            fl = obj.get("failure_label", "")
            if fl:
                failure_log.append({
                    "ts":      ts_str,
                    "station": station,
                    "label":   fl,
                })

            # State machine: update current step label
            import re as _re
            sub_or_task = (obj.get("current_sub_task") or obj.get("current_task") or "").lower()
            resolved = "idle"
            for pattern, label in STEP_KEYWORDS.get(station, []):
                if pattern and _re.search(pattern, sub_or_task):
                    resolved = label
                    break
                elif not pattern and sub_or_task:
                    resolved = label
                    break
            prev_label = sm_state[station]["step_label"]
            sm_state[station].update({
                "step_label": resolved,
                "task":    obj.get("current_task", ""),
                "sub":     obj.get("current_sub_task", ""),
                "elapsed": obj.get("current_task_duration", 0.0),
                "state":   obj.get("current_state", ""),
            })
            if resolved != prev_label and resolved != "idle":
                hist = sm_state[station]["history"]
                if not hist or hist[-1] != resolved:
                    hist.append(resolved)
                    if len(hist) > 12:
                        hist.pop(0)

            # Gantt: track sub-task transitions
            cur_task = obj.get("current_task") or ""
            cur_sub  = obj.get("current_sub_task") or ""
            active   = _active_sub[station]
            # Use sub_task if present, else fall back to task name
            label = cur_sub if cur_sub else cur_task
            if label != active["sub"] or cur_task != active["task"]:
                # Close the previous block
                if active["sub"] and active["start"] is not None:
                    gantt_tasks.append({
                        "station":  station,
                        "task":     active["task"],
                        "sub_task": active["sub"],
                        "start":    active["start"],
                        "end":      ts,
                    })
                # Open a new block
                _active_sub[station] = {
                    "task":  cur_task,
                    "sub":   label,
                    "start": ts if label else None,
                }

    playback["running"] = False


# ── DASH APP ─────────────────────────────────────────────────────────────────

app = Dash(__name__, title="Smart Factory Dashboard")
app.config.suppress_callback_exceptions = True

# ── LAYOUT HELPERS ───────────────────────────────────────────────────────────

def make_indicator(label, value, color="#10b981"):
    """Small KPI card."""
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#94a3b8",
                                "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        html.Div(str(value), style={"fontSize": "20px", "fontWeight": "600",
                                     "color": color, "marginTop": "2px"}),
    ], style={"background": "#1e293b", "borderRadius": "8px",
              "padding": "10px 14px", "minWidth": "110px"})


def bool_badge(val):
    on = bool(val)
    color = "#10b981" if on else "#475569"
    txt = "ON" if on else "OFF"
    return html.Span(txt, style={
        "background": color, "color": "#fff", "borderRadius": "4px",
        "padding": "2px 8px", "fontSize": "11px", "fontWeight": "600",
        "marginLeft": "6px",
    })


def sensor_row(label, val):
    is_bool = isinstance(val, bool)
    is_num  = isinstance(val, (int, float)) and not isinstance(val, bool)
    return html.Div([
        html.Span(label, style={"fontSize": "12px", "color": "#cbd5e1",
                                 "fontFamily": "monospace", "minWidth": "180px",
                                 "display": "inline-block"}),
        bool_badge(val) if is_bool else
        html.Span(f"{val:.2f}" if isinstance(val, float) else str(val),
                  style={"color": "#f8fafc", "fontSize": "13px",
                         "fontWeight": "500", "marginLeft": "6px"}),
    ], style={"display": "flex", "alignItems": "center",
              "padding": "4px 0", "borderBottom": "1px solid #1e293b"})


# ── MAIN LAYOUT ──────────────────────────────────────────────────────────────

app.layout = html.Div(style={
    "background": "#0f172a", "minHeight": "100vh",
    "fontFamily": "'Inter', 'Segoe UI', sans-serif", "color": "#f1f5f9",
    "padding": "16px 20px",
}, children=[

    # Header
    html.Div([
        html.Div([
            html.H1("Fischertechnik Smart Factory",
                    style={"margin": 0, "fontSize": "22px", "fontWeight": "600"}),
            html.Div("Live Playback Dashboard · University of St.Gallen",
                     style={"fontSize": "13px", "color": "#64748b", "marginTop": "2px"}),
        ]),
        html.Div([
            html.Button("▶  Play", id="btn-play",
                        style={"background": "#10b981", "color": "#fff",
                               "border": "none", "borderRadius": "6px",
                               "padding": "8px 18px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": "600",
                               "marginRight": "8px"}),
            html.Button("⏸  Pause", id="btn-pause",
                        style={"background": "#334155", "color": "#fff",
                               "border": "none", "borderRadius": "6px",
                               "padding": "8px 18px", "cursor": "pointer",
                               "fontSize": "13px", "marginRight": "8px"}),
            html.Div([
                html.Span("Speed", style={"fontSize": "12px", "color": "#64748b",
                                          "marginRight": "8px"}),
                dcc.Slider(id="speed-slider", min=1, max=50, step=1, value=5,
                           marks={1: "1×", 10: "10×", 25: "25×", 50: "50×"},
                           tooltip={"placement": "top"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "center", "marginBottom": "16px",
              "borderBottom": "1px solid #1e293b", "paddingBottom": "12px"}),

    # Progress bar
    html.Div([
        html.Div(id="progress-bar-fill",
                 style={"height": "4px", "background": "#6366f1",
                        "borderRadius": "2px", "width": "0%",
                        "transition": "width 0.3s ease"}),
    ], style={"background": "#1e293b", "borderRadius": "2px",
              "height": "4px", "marginBottom": "12px"}),

    html.Div(id="progress-label",
             style={"fontSize": "11px", "color": "#64748b",
                    "marginBottom": "16px", "textAlign": "right"}),

    # Top KPI row
    html.Div(id="kpi-row",
             style={"display": "flex", "gap": "10px",
                    "flexWrap": "wrap", "marginBottom": "20px"}),

    # Station tabs
    dcc.Tabs(id="station-tabs", value="MM_1",
             style={"marginBottom": "16px"},
             colors={"border": "#1e293b", "primary": "#6366f1",
                     "background": "#0f172a"},
             children=[
                 dcc.Tab(label=STATION_LABELS[s], value=s,
                         style={"color": "#64748b", "background": "#0f172a",
                                "border": "none", "padding": "8px 14px",
                                "fontSize": "13px"},
                         selected_style={"color": STATION_COLORS[s],
                                         "background": "#1e293b",
                                         "border": f"2px solid {STATION_COLORS[s]}",
                                         "borderBottom": "none",
                                         "fontWeight": "600"})
                 for s in STATIONS
             ]),

    # Station content area
    html.Div(id="station-content"),

    # Charts row
    html.Div([
        # Motor speeds chart
        html.Div([
            html.Div("Motor Speeds", style={"fontSize": "13px", "fontWeight": "600",
                                            "color": "#94a3b8", "marginBottom": "8px"}),
            dcc.Graph(id="motor-chart", style={"height": "220px"},
                      config={"displayModeBar": False}),
        ], style={"background": "#1e293b", "borderRadius": "10px",
                  "padding": "14px", "flex": "1", "minWidth": "320px"}),

        # Position chart
        html.Div([
            html.Div("Position Tracking", style={"fontSize": "13px", "fontWeight": "600",
                                                  "color": "#94a3b8", "marginBottom": "8px"}),
            dcc.Graph(id="pos-chart", style={"height": "220px"},
                      config={"displayModeBar": False}),
        ], style={"background": "#1e293b", "borderRadius": "10px",
                  "padding": "14px", "flex": "1", "minWidth": "320px"}),
    ], style={"display": "flex", "gap": "14px",
              "flexWrap": "wrap", "marginTop": "16px"}),

    # State machine panel
    html.Div([
        html.Div("State machine — current step per station",
                 style={"fontSize": "13px", "fontWeight": "600",
                        "color": "#94a3b8", "marginBottom": "12px"}),
        html.Div(id="state-machine-panel"),
    ], style={"background": "#1e293b", "borderRadius": "10px",
              "padding": "16px", "marginTop": "16px"}),

    # Failure log
    html.Div([
        html.Div("⚠  Anomaly / Failure Log",
                 style={"fontSize": "13px", "fontWeight": "600",
                        "color": "#ef4444", "marginBottom": "8px"}),
        html.Div(id="failure-log-content",
                 style={"maxHeight": "160px", "overflowY": "auto",
                        "fontSize": "12px"}),
    ], style={"background": "#1e293b", "borderRadius": "10px",
              "padding": "14px", "marginTop": "14px"}),

    # Auto-refresh
    dcc.Interval(id="interval", interval=500, n_intervals=0),
    dcc.Store(id="selected-station", data="MM_1"),
])


# ── CALLBACKS ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("progress-bar-fill", "style"),
    Output("progress-label", "children"),
    Input("interval", "n_intervals"),
)
def update_progress(_):
    total = playback["total_lines"]
    current = playback["current_line"]
    pct = (current / total * 100) if total else 0
    status = "▶ Playing" if (playback["running"] and not playback["paused"]) else \
             ("⏸ Paused" if playback["paused"] else "⏹ Stopped")
    label = f"{status}  ·  {current:,} / {total:,} readings  ·  {pct:.1f}%  ·  Speed {playback['speed']}×"
    style = {"height": "4px", "background": "#6366f1",
             "borderRadius": "2px", "width": f"{pct:.1f}%",
             "transition": "width 0.4s ease"}
    return style, label


@app.callback(
    Output("kpi-row", "children"),
    Input("interval", "n_intervals"),
)
def update_kpi(_):
    cards = []
    for s in STATIONS:
        d = latest.get(s, {})
        state = d.get("current_state", "—")
        color = "#10b981" if state == "ready" else \
                "#f59e0b" if state == "working" else \
                "#ef4444" if state == "error" else "#64748b"
        cards.append(html.Div([
            html.Div(s, style={"fontSize": "11px", "color": "#64748b",
                               "fontWeight": "600", "letterSpacing": "0.04em"}),
            html.Div(state, style={"fontSize": "14px", "fontWeight": "600",
                                   "color": color, "marginTop": "2px"}),
            html.Div(d.get("current_task", "") or "idle",
                     style={"fontSize": "10px", "color": "#475569",
                            "marginTop": "2px", "overflow": "hidden",
                            "textOverflow": "ellipsis", "whiteSpace": "nowrap",
                            "maxWidth": "130px"}),
        ], style={"background": "#1e293b", "borderRadius": "8px",
                  "padding": "10px 14px", "minWidth": "130px",
                  "borderLeft": f"3px solid {STATION_COLORS[s]}"}))
    return cards


@app.callback(
    Output("station-content", "children"),
    Input("interval", "n_intervals"),
    Input("station-tabs", "value"),
)
def update_station_content(_, station):
    d = latest.get(station, {})
    if not d:
        return html.Div("Waiting for data...",
                        style={"color": "#475569", "padding": "20px"})

    # Separate fields into sensors, actuators, process
    sensors, actuators, process_fields = [], [], []
    skip = {"id", "station", "timestamp", "current_stock"}
    for k, v in d.items():
        if k in skip:
            continue
        if k.startswith("i"):
            sensors.append((k, v))
        elif k.startswith(("m", "o")):
            actuators.append((k, v))
        elif k.startswith("current") or k == "failure_label":
            process_fields.append((k, v))
        elif "pos" in k or "target" in k:
            process_fields.append((k, v))

    def section(title, color, items):
        return html.Div([
            html.Div(title, style={"fontSize": "11px", "fontWeight": "600",
                                   "color": color, "textTransform": "uppercase",
                                   "letterSpacing": "0.05em",
                                   "marginBottom": "6px"}),
            html.Div([sensor_row(k, v) for k, v in items]),
        ], style={"background": "#1e293b", "borderRadius": "8px",
                  "padding": "12px 14px", "flex": "1", "minWidth": "220px"})

    rows = [
        section("Sensors", "#0ea5e9", sensors),
        section("Actuators", "#f59e0b", actuators),
        section("Process State", "#10b981", process_fields),
    ]

    # HBW stock grid
    if station == "HBW_1" and "current_stock" in d:
        stock = d["current_stock"]
        cells = []
        for i in range(9):
            val = stock.get(str(i), "")
            occupied = bool(val)
            cells.append(html.Div([
                html.Div(f"Slot {i}", style={"fontSize": "10px", "color": "#64748b"}),
                html.Div(val if occupied else "empty",
                         style={"fontSize": "12px", "fontWeight": "600",
                                "color": "#10b981" if occupied else "#475569",
                                "marginTop": "2px"}),
            ], style={"background": "#0f172a", "borderRadius": "6px",
                      "padding": "8px", "textAlign": "center",
                      "border": f"1px solid {'#10b981' if occupied else '#1e293b'}"}))
        rows.append(html.Div([
            html.Div("Warehouse Stock", style={"fontSize": "11px", "fontWeight": "600",
                                               "color": "#f59e0b",
                                               "textTransform": "uppercase",
                                               "letterSpacing": "0.05em",
                                               "marginBottom": "8px"}),
            html.Div(cells, style={"display": "grid",
                                   "gridTemplateColumns": "repeat(3, 1fr)",
                                   "gap": "6px"}),
        ], style={"background": "#1e293b", "borderRadius": "8px",
                  "padding": "12px 14px", "minWidth": "220px"}))

    ts = d.get("timestamp", "")
    return html.Div([
        html.Div(f"Last update: {ts}", style={"fontSize": "11px", "color": "#475569",
                                              "marginBottom": "10px"}),
        html.Div(rows, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ])


@app.callback(
    Output("motor-chart", "figure"),
    Input("interval", "n_intervals"),
    Input("station-tabs", "value"),
)
def update_motor_chart(_, station):
    mh = motor_history.get(station, {})
    fig = go.Figure()
    colors = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b"]
    for i, (motor, hist) in enumerate(mh.items()):
        if not hist:
            continue
        ts_vals = [h[0] for h in hist]
        speeds   = [h[1] for h in hist]
        fig.add_trace(go.Scatter(
            x=ts_vals, y=speeds, mode="lines",
            name=motor, line={"color": colors[i % len(colors)], "width": 1.5},
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8", "size": 11},
        margin={"l": 40, "r": 10, "t": 10, "b": 30},
        legend={"orientation": "h", "y": 1.1},
        xaxis={"gridcolor": "#1e293b", "showgrid": True},
        yaxis={"gridcolor": "#1e293b", "showgrid": True, "title": "Speed"},
    )
    return fig


@app.callback(
    Output("pos-chart", "figure"),
    Input("interval", "n_intervals"),
    Input("station-tabs", "value"),
)
def update_pos_chart(_, station):
    fig = go.Figure()

    if station == "VGR_1":
        ph = pos_history["VGR_1"]
        ts = list(ph["ts"])
        for axis, color in [("x", "#6366f1"), ("y", "#0ea5e9"), ("z", "#10b981")]:
            vals = list(ph[axis])
            if vals:
                fig.add_trace(go.Scatter(
                    x=ts, y=vals, mode="lines", name=axis.upper(),
                    line={"color": color, "width": 1.5},
                ))
    elif station == "HBW_1":
        ph = pos_history["HBW_1"]
        ts = list(ph["ts"])
        for axis, color in [("x", "#6366f1"), ("y", "#0ea5e9")]:
            vals = list(ph[axis])
            if vals:
                fig.add_trace(go.Scatter(
                    x=ts, y=vals, mode="lines", name=axis.upper(),
                    line={"color": color, "width": 1.5},
                ))
    else:
        fig.add_annotation(text=f"No position data for {station}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"color": "#475569", "size": 13})

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8", "size": 11},
        margin={"l": 40, "r": 10, "t": 10, "b": 30},
        legend={"orientation": "h", "y": 1.1},
        xaxis={"gridcolor": "#1e293b", "showgrid": True},
        yaxis={"gridcolor": "#1e293b", "showgrid": True, "title": "Position"},
    )
    return fig



@app.callback(
    Output("state-machine-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_state_machine(_):
    rows = []
    for s in STATIONS:
        sm  = sm_state[s]
        col = STATION_COLORS[s]
        steps = STATION_STEPS[s]
        cur   = sm["step_label"]
        hist  = sm["history"]
        task  = sm["task"] or "—"
        elapsed = sm["elapsed"]
        state_str = sm["state"]

        # Shorten task label
        task_short = task if len(task) <= 55 else task[:53] + "..."

        # Build step pills
        pills = []
        for step in steps:
            is_active = (step == cur)
            is_done   = (step in hist and not is_active)
            if is_active:
                bg, border, txt = col + "33", col, col
                weight = "600"
            elif is_done:
                bg, border, txt = "#1e293b", "#334155", "#64748b"
                weight = "400"
            else:
                bg, border, txt = "transparent", "#1e293b", "#334155"
                weight = "400"

            pills.append(html.Div(step, style={
                "padding": "3px 10px",
                "borderRadius": "4px",
                "fontSize": "11px",
                "fontWeight": weight,
                "background": bg,
                "border": f"1px solid {border}",
                "color": txt,
                "whiteSpace": "nowrap",
                "transition": "all 0.3s ease",
            }))
            # Arrow between pills (not after last)
            if step != steps[-1]:
                pills.append(html.Span("→", style={
                    "color": "#334155", "fontSize": "11px",
                    "padding": "0 2px", "flexShrink": "0",
                }))

        # Elapsed badge
        elapsed_badge = html.Span(
            f"{elapsed:.1f}s" if elapsed else "",
            style={"fontSize": "10px", "color": "#475569",
                   "marginLeft": "10px", "fontFamily": "monospace"}
        )

        rows.append(html.Div([
            # Station label + state dot
            html.Div([
                html.Div(style={
                    "width": "8px", "height": "8px", "borderRadius": "50%",
                    "background": col if state_str not in ("ready", "") else "#334155",
                    "flexShrink": "0", "marginTop": "2px",
                }),
                html.Div([
                    html.Span(s, style={"fontSize": "12px", "fontWeight": "600",
                                        "color": col, "fontFamily": "monospace"}),
                    html.Div(task_short, style={"fontSize": "10px", "color": "#475569",
                                                "marginTop": "1px", "maxWidth": "160px",
                                                "overflow": "hidden", "textOverflow": "ellipsis",
                                                "whiteSpace": "nowrap"}),
                ], style={"marginLeft": "8px"}),
            ], style={"display": "flex", "alignItems": "flex-start",
                      "width": "180px", "flexShrink": "0"}),

            # Step pills row
            html.Div(pills + [elapsed_badge], style={
                "display": "flex", "alignItems": "center",
                "flexWrap": "wrap", "gap": "3px", "flex": "1",
            }),
        ], style={
            "display": "flex", "alignItems": "center",
            "padding": "10px 0",
            "borderBottom": "1px solid #1e293b" if s != STATIONS[-1] else "none",
            "gap": "12px",
        }))

    return rows



@app.callback(
    Output("failure-log-content", "children"),
    Input("interval", "n_intervals"),
)
def update_failure_log(_):
    if not failure_log:
        return html.Div("No failures detected yet.",
                        style={"color": "#475569", "fontSize": "12px",
                               "padding": "8px 0"})
    rows = []
    for entry in reversed(list(failure_log)):
        rows.append(html.Div([
            html.Span(entry["ts"][-12:],
                      style={"color": "#64748b", "marginRight": "8px"}),
            html.Span(entry["station"],
                      style={"color": STATION_COLORS.get(entry["station"], "#fff"),
                             "fontWeight": "600", "marginRight": "8px"}),
            html.Span(entry["label"], style={"color": "#ef4444"}),
        ], style={"padding": "4px 0", "borderBottom": "1px solid #0f172a"}))
    return rows


@app.callback(
    Output("interval", "disabled"),
    Input("btn-play",  "n_clicks"),
    Input("btn-pause", "n_clicks"),
    prevent_initial_call=True,
)
def control_playback(play_clicks, pause_clicks):
    ctx = callback_context.triggered[0]["prop_id"]
    if "btn-play" in ctx:
        if not playback["running"]:
            playback["running"] = True
            playback["paused"]  = False
            t = threading.Thread(
                target=run_playback,
                args=(playback["log_file"],),
                daemon=True,
            )
            t.start()
        else:
            playback["paused"] = False
        return False
    elif "btn-pause" in ctx:
        playback["paused"] = True
        return False
    return False


@app.callback(
    Output("speed-slider", "value"),
    Input("speed-slider", "value"),
)
def update_speed(val):
    playback["speed"] = float(val)
    return val


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Factory Live Playback Dashboard")
    parser.add_argument(
        "--file", "-f",
        default="low-level_log_20230206-140808.txt",
        help="Path to the log file",
    )
    parser.add_argument("--port", "-p", type=int, default=8050)
    args = parser.parse_args()

    log_path = Path(args.file)
    if not log_path.exists():
        print(f"ERROR: File not found: {log_path}")
        exit(1)

    playback["log_file"] = str(log_path)

    print(f"""
╔══════════════════════════════════════════════════════╗
║   Fischertechnik Smart Factory Dashboard             ║
║   Log file : {log_path.name:<40}║
║   Open     : http://127.0.0.1:{args.port}                    ║
║   Press    : Play in the browser to start            ║
╚══════════════════════════════════════════════════════╝
""")
    app.run(debug=False, port=args.port)
