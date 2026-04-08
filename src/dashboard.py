"""
Fischertechnik Smart Factory — Black & White Predictive Health Dashboard
Features:
  - Black & white technological theme
  - Predictive health (trend-based failure risk)
  - Real-time alert system (critical threshold < 50%)
  - Threshold-based logic (health degradation + motor checks)
  - Time-based updates via interval loop
  - Root-cause foundation (failure + motor influence on health)
  - KPI aggregation per machine
  - Statistical anomaly detection framework
"""

import json
import argparse
import threading
import time
import collections
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback_context, State
from dash.exceptions import PreventUpdate

# ─── MACHINE IDENTITY ────────────────────────────────────────────────────────
STATIONS = ["MM_1", "HBW_1", "VGR_1", "SM_1", "OV_1", "WT_1", "EC_1"]

MACHINE_NAMES = {
    "MM_1":  "Milling Machine",
    "HBW_1": "Automated Warehouse",
    "VGR_1": "Robot Arm (VGR)",
    "SM_1":  "Sorting Machine",
    "OV_1":  "Oven",
    "WT_1":  "Transfer Unit",
    "EC_1":  "Env. Controller",
}

STATION_ICONS = {
    "MM_1":  "[MM]",
    "HBW_1": "[HBW]",
    "VGR_1": "[VGR]",
    "SM_1":  "[SM]",
    "OV_1":  "[OV]",
    "WT_1":  "[WT]",
    "EC_1":  "[EC]",
}

PRODUCTION_STATIONS = ["MM_1", "OV_1", "SM_1"]
LOGISTICS_STATIONS  = ["VGR_1", "HBW_1", "WT_1"]

MAX_SPEED = 2000

# ─── SHARED STATE ─────────────────────────────────────────────────────────────
latest       = {s: {} for s in STATIONS}
history      = {s: collections.deque(maxlen=300) for s in STATIONS}
motor_history = {s: collections.defaultdict(lambda: collections.deque(maxlen=300))
                 for s in STATIONS}
pos_history  = {
    "VGR_1": {"x": collections.deque(maxlen=300), "y": collections.deque(maxlen=300),
               "z": collections.deque(maxlen=300), "ts": collections.deque(maxlen=300)},
    "HBW_1": {"x": collections.deque(maxlen=300), "y": collections.deque(maxlen=300),
               "ts": collections.deque(maxlen=300)},
}
failure_log  = collections.deque(maxlen=200)
gantt_tasks  = collections.deque(maxlen=600)
_active_sub  = {s: {"task": None, "sub": None, "start": None} for s in STATIONS}

# ─── ALERT SYSTEM ─────────────────────────────────────────────────────────────
CRITICAL_HEALTH_THRESHOLD = 50.0   # triggers red alert
WARNING_HEALTH_THRESHOLD  = 70.0   # triggers yellow alert
MOTOR_DEVIATION_THRESHOLD = 0.40   # 40% deviation → anomaly
MOTOR_STALL_THRESHOLD     = 30.0   # baseline RPM below which stall is meaningful
DURATION_RATIO_WARNING    = 1.4    # 40% longer than baseline → warning
DURATION_RATIO_CRITICAL   = 2.0    # 2× longer → critical
MOTOR_BASELINE_MIN_SAMPLES = 12
DURATION_BASELINE_MIN_SAMPLES = 8
ALERT_COOLDOWN_SECONDS = {"critical": 25, "warning": 45, "info": 30}

alerts_lock  = threading.Lock()
active_alerts = collections.deque(maxlen=100)   # {ts, station, severity, msg}
alert_last_seen = {}
alert_band_state = {
    s: {"health": "nominal", "risk": "low"} for s in STATIONS
}
motor_anomaly_streak = {
    s: {mk: 0 for mk in ("m1_speed", "m2_speed", "m3_speed", "m4_speed")} for s in STATIONS
}
duration_anomaly_streak = {s: 0 for s in STATIONS}

def _push_alert(station: str, severity: str, msg: str, ts: datetime, category=None):
    """severity: 'critical' | 'warning' | 'info'"""
    with alerts_lock:
        cooldown_key = (station, category or msg, severity)
        last_ts = alert_last_seen.get(cooldown_key)
        if last_ts is not None:
            if (ts - last_ts).total_seconds() < ALERT_COOLDOWN_SECONDS.get(severity, 30):
                return
        # deduplicate: don't push same msg twice in 10 seconds
        now_ts = ts
        for a in reversed(active_alerts):
            if a["station"] == station and a["msg"] == msg:
                diff = (now_ts - a["ts"]).total_seconds()
                if diff < 10:
                    return
                break
        alert_last_seen[cooldown_key] = ts
        active_alerts.append({"ts": ts, "station": station, "severity": severity, "msg": msg})

# ─── HEALTH / DETERIORATION STATE ────────────────────────────────────────────
MAX_HEALTH_HISTORY = 600

health_lock        = threading.Lock()
health_scores      = {s: 100.0 for s in STATIONS}
health_trend       = {s: collections.deque(maxlen=MAX_HEALTH_HISTORY) for s in STATIONS}
health_components  = {s: {"fault": 100.0, "motor": 100.0, "duration": 100.0}
                      for s in STATIONS}

# Predictive state
prediction_lock    = threading.Lock()
predicted_rul      = {s: None for s in STATIONS}   # remaining useful life estimate (%)
failure_risk       = {s: 0.0  for s in STATIONS}   # 0–100 probability estimate
anomaly_score      = {s: 0.0  for s in STATIONS}   # z-score based anomaly indicator
kpi_store          = {s: {"faults_total": 0, "stalls": 0, "duration_exceedances": 0,
                           "motor_anomalies": 0, "uptime_pct": 100.0,
                           "avg_task_dur": 0.0, "tasks_completed": 0}
                      for s in STATIONS}

_duration_baseline = {s: collections.deque(maxlen=50) for s in STATIONS}
_motor_baseline    = {
    s: {mk: collections.deque(maxlen=60) for mk in ("m1_speed", "m2_speed", "m3_speed", "m4_speed")}
    for s in STATIONS
}
_fault_penalty     = {s: 0.0 for s in STATIONS}

# ─── STATISTICAL ANOMALY DETECTION ───────────────────────────────────────────
_anomaly_buffers   = {s: collections.deque(maxlen=100) for s in STATIONS}
_anomaly_stats     = {s: {"mean": 0.0, "std": 1.0, "n": 0} for s in STATIONS}

def _update_anomaly_detector(station: str, value: float, ts: datetime):
    """Welford online mean/variance, output Z-score based anomaly score 0-100."""
    buf = _anomaly_buffers[station]
    buf.append(value)
    if len(buf) < 5:
        return
    mean_v = statistics.mean(buf)
    std_v  = statistics.pstdev(buf) or 1.0
    z      = abs(value - mean_v) / std_v
    # Map z-score to 0-100 anomaly score (z>3 → near 100)
    score  = min(100.0, (z / 4.0) * 100.0)
    with prediction_lock:
        anomaly_score[station] = round(score, 1)


def _predict_rul(station: str):
    """
    Simple linear regression on recent health trend.
    Extrapolate when health hits 0 → Remaining Useful Life in % of trajectory.
    Returns: (rul_pct, risk_pct)
    """
    with health_lock:
        trend = list(health_trend[station])
    if len(trend) < 10:
        return None, 0.0

    # Use last N points
    window = trend[-60:] if len(trend) >= 60 else trend
    xs = list(range(len(window)))
    ys = [t[1] for t in window]

    n   = len(xs)
    sx  = sum(xs)
    sy  = sum(ys)
    sxy = sum(x*y for x, y in zip(xs, ys))
    sxx = sum(x*x for x in xs)
    denom = (n * sxx - sx * sx)
    if denom == 0:
        return None, 0.0
    slope     = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    current_y = intercept + slope * (n - 1)

    # Risk: steeper negative slope = higher risk
    if slope >= 0:
        risk = 0.0
        rul  = 100.0
    else:
        # Steps to reach 0 from current position at this slope
        if current_y <= 0:
            steps_to_zero = 0
        else:
            steps_to_zero = current_y / abs(slope)
        rul  = min(100.0, (steps_to_zero / max(len(window), 1)) * 100.0)
        risk = max(0.0, min(100.0, 100.0 - rul))

    return round(rul, 1), round(risk, 1)

def _health_band(score: float) -> str:
    if score < CRITICAL_HEALTH_THRESHOLD:
        return "critical"
    if score < WARNING_HEALTH_THRESHOLD:
        return "warning"
    return "nominal"

def _risk_band(risk: float) -> str:
    if risk > 75:
        return "high"
    if risk > 45:
        return "elevated"
    return "low"

def _baseline_ready(buf, min_samples: int) -> bool:
    return len(buf) >= min_samples

def _robust_center(values) -> float:
    vals = sorted(list(values))
    if not vals:
        return 0.0
    return vals[len(vals) // 2]


def _compute_health(station: str, obj: dict, ts: datetime):
    with health_lock:
        comp = health_components[station]
        learn_baseline = not obj.get("failure_label") and obj.get("current_state", "") != "error"

        # ── 1. FAULT SCORE ────────────────────────────────────────────────
        fl = obj.get("failure_label", "")
        if fl:
            _fault_penalty[station] = min(100.0, _fault_penalty[station] + 15.0)
            kpi_store[station]["faults_total"] += 1
            _push_alert(station, "critical", f"Fault: {fl}", ts, category="fault")
        else:
            _fault_penalty[station] = max(0.0, _fault_penalty[station] - 0.3)
        comp["fault"] = max(0.0, 100.0 - _fault_penalty[station])

        # ── 2. MOTOR SCORE ────────────────────────────────────────────────
        motor_penalties = []
        for mk in ("m1_speed", "m2_speed", "m3_speed", "m4_speed"):
            if mk not in obj:
                continue
            raw   = abs(float(obj[mk]))
            bq    = _motor_baseline[station][mk]
            if _baseline_ready(bq, MOTOR_BASELINE_MIN_SAMPLES):
                mean_base = _robust_center(bq)
                if mean_base > 10:
                    if raw == 0 and mean_base > MOTOR_STALL_THRESHOLD:
                        motor_anomaly_streak[station][mk] += 1
                        if motor_anomaly_streak[station][mk] >= 2:
                            motor_penalties.append(25.0)
                            kpi_store[station]["stalls"] += 1
                            _push_alert(station, "warning", f"Motor stall detected: {mk}", ts,
                                        category=f"motor_stall_{mk}")
                    else:
                        dev = abs(raw - mean_base) / max(mean_base, 1)
                        if dev > MOTOR_DEVIATION_THRESHOLD:
                            motor_anomaly_streak[station][mk] += 1
                            if motor_anomaly_streak[station][mk] >= 3 or dev > 0.85:
                                penalty = min(20.0, dev * 20)
                                motor_penalties.append(penalty)
                                kpi_store[station]["motor_anomalies"] += 1
                                _push_alert(station, "warning",
                                            f"Motor anomaly {mk}: {dev*100:.0f}% deviation", ts,
                                            category=f"motor_dev_{mk}")
                        else:
                            motor_anomaly_streak[station][mk] = 0
                else:
                    motor_anomaly_streak[station][mk] = 0
            else:
                motor_anomaly_streak[station][mk] = 0
            if learn_baseline and raw > 0:
                bq.append(raw)

            # Feed anomaly detector with motor speed
            _update_anomaly_detector(station, raw, ts)

        if motor_penalties:
            comp["motor"] = max(0.0, comp["motor"] - min(50.0, sum(motor_penalties)) * 0.4)
        else:
            comp["motor"] = min(100.0, comp["motor"] + 0.2)

        # ── 3. DURATION SCORE ─────────────────────────────────────────────
        dur = obj.get("current_task_duration", None)
        if dur is not None and float(dur) > 0.5:
            dur = float(dur)
            bq  = _duration_baseline[station]
            if _baseline_ready(bq, DURATION_BASELINE_MIN_SAMPLES):
                median_dur = _robust_center(bq)
                if median_dur > 0:
                    ratio = dur / median_dur
                    if ratio > DURATION_RATIO_CRITICAL:
                        duration_anomaly_streak[station] += 1
                        comp["duration"] = max(0.0, comp["duration"] - min(30.0, (ratio-1)*20)*0.3)
                        kpi_store[station]["duration_exceedances"] += 1
                        if duration_anomaly_streak[station] >= 2 or ratio > 2.5:
                            _push_alert(station, "warning",
                                        f"Task duration {ratio:.1f}× baseline ({dur:.1f}s)", ts,
                                        category="task_duration")
                    elif ratio > DURATION_RATIO_WARNING:
                        duration_anomaly_streak[station] += 1
                        comp["duration"] = max(0.0, comp["duration"] - min(15.0, (ratio-1)*10)*0.2)
                    else:
                        duration_anomaly_streak[station] = 0
                        comp["duration"] = min(100.0, comp["duration"] + 0.15)
            else:
                duration_anomaly_streak[station] = 0
            if learn_baseline and (not _baseline_ready(bq, DURATION_BASELINE_MIN_SAMPLES)
                                   or duration_anomaly_streak[station] == 0):
                bq.append(dur)
            # rolling avg task duration
            kpi_store[station]["tasks_completed"] += 1
            n   = kpi_store[station]["tasks_completed"]
            old = kpi_store[station]["avg_task_dur"]
            kpi_store[station]["avg_task_dur"] = old + (dur - old) / n

        # ── COMPOSITE HEALTH ──────────────────────────────────────────────
        score = (comp["fault"] * 0.45 + comp["motor"] * 0.30 + comp["duration"] * 0.25)
        health_scores[station] = round(score, 1)
        health_trend[station].append((ts, score))

        current_health_band = _health_band(score)
        previous_health_band = alert_band_state[station]["health"]
        if current_health_band != previous_health_band:
            if current_health_band == "critical":
                _push_alert(station, "critical",
                            f"Health CRITICAL: {score:.0f}% (< {CRITICAL_HEALTH_THRESHOLD:.0f}%)", ts,
                            category="health_band")
            elif current_health_band == "warning":
                _push_alert(station, "warning",
                            f"Health WARNING: {score:.0f}% (< {WARNING_HEALTH_THRESHOLD:.0f}%)", ts,
                            category="health_band")
            else:
                _push_alert(station, "info",
                            f"Health recovered to nominal: {score:.0f}%", ts,
                            category="health_band")
            alert_band_state[station]["health"] = current_health_band

    # Predictive RUL (outside health_lock to avoid deadlock)
    rul, risk = _predict_rul(station)
    with prediction_lock:
        predicted_rul[station]  = rul
        failure_risk[station]   = risk
        current_risk_band = _risk_band(risk)
        previous_risk_band = alert_band_state[station]["risk"]
        if current_risk_band != previous_risk_band:
            if current_risk_band == "high":
                _push_alert(station, "critical",
                            f"Failure risk HIGH: {risk:.0f}% — immediate inspection", ts,
                            category="risk_band")
            elif current_risk_band == "elevated":
                _push_alert(station, "warning",
                            f"Failure risk ELEVATED: {risk:.0f}%", ts,
                            category="risk_band")
            else:
                _push_alert(station, "info",
                            f"Failure risk returned to low: {risk:.0f}%", ts,
                            category="risk_band")
            alert_band_state[station]["risk"] = current_risk_band

# ─── EC STATE ─────────────────────────────────────────────────────────────────
MAX_EC     = 400
ec_lock    = threading.Lock()
ec_latest  = {}
ec_history = {k: collections.deque(maxlen=MAX_EC)
              for k in ("mean", "variance", "rms", "max", "min", "ber", "state", "ts")}
ec_modulation_counts = collections.defaultdict(int)

def update_ec_state(obj: dict, ts: datetime):
    if obj.get("station") not in ("EC", "EC_1"):
        return
    with ec_lock:
        ec_latest.update(obj)
        feats = obj.get("features", {})
        ec_history["ts"].append(ts)
        ec_history["ber"].append(float(obj.get("ber", 0.0)))
        ec_history["state"].append(int(obj.get("state", 0)))
        for key in ("mean", "variance", "rms", "max", "min"):
            ec_history[key].append(float(feats.get(key, 0.0)))
        ec_modulation_counts[obj.get("modulation", "UNKNOWN")] += 1

# ─── STATE MACHINE ────────────────────────────────────────────────────────────
STATION_STEPS = {
    "MM_1":  ["idle", "transport to mill", "milling", "eject", "transport to sorter", "return home"],
    "HBW_1": ["idle", "move to slot", "pick up bucket", "transport to belt", "drop off",
               "wait for VGR", "transport to crane jib", "pick up from belt", "transport to slot", "drop off at slot"],
    "VGR_1": ["idle", "calibrate", "move to pickup", "pick up workpiece", "transport", "drop off"],
    "SM_1":  ["idle", "detect color", "transport to sink", "eject to sink"],
    "OV_1":  ["idle", "move to oven", "open door", "load workpiece", "close door", "burning", "unload"],
    "WT_1":  ["idle", "conveyor moving", "valve active", "transfer done"],
    "EC_1":  ["idle", "monitoring"],
}

STEP_KEYWORDS = {
    "MM_1":  [("transport.*mill","transport to mill"),("milling","milling"),
              ("eject","eject"),("transport.*sort","transport to sorter"),
              ("ejection","eject"),("initial","return home")],
    "HBW_1": [("moving towards","move to slot"),("picking up.*slot","pick up bucket"),
              ("transport.*conveyor","transport to belt"),("dropping.*conveyor","drop off"),
              ("waiting","wait for VGR"),("transport.*crane","transport to crane jib"),
              ("picking up.*conveyor","pick up from belt"),("transport.*slot","transport to slot"),
              ("dropping.*slot","drop off at slot")],
    "VGR_1": [("calibrat","calibrate"),("moving towards","move to pickup"),
              ("picking up","pick up workpiece"),("transport","transport"),
              ("dropping off","drop off")],
    "SM_1":  [("detect","detect color"),("transport.*sink","transport to sink"),("eject","eject to sink")],
    "OV_1":  [("moving towards","move to oven"),("open.*door","open door"),
              ("transport.*inside","load workpiece"),("close.*door|closing","close door"),
              ("burning","burning"),("transport.*outside","unload"),
              ("picking up","pick up"),("dropping off","drop off")],
    "WT_1":  [("conveyor|motor","conveyor moving"),("valve","valve active")],
    "EC_1":  [("","monitoring")],
}

sm_state = {s: {"step_label": "idle", "task": "", "sub": "", "elapsed": 0.0,
                "state": "ready", "history": []} for s in STATIONS}

# ─── ENGINEERING DESIGN TOKENS ───────────────────────────────────────────────
BG      = "#08111d"
BG2     = "#0d1724"
CARD    = "#101b2a"
CARD2   = "#142235"
CARD3   = "#1a2c42"
GRID    = "#24364a"
GRID2   = "#34506d"
MUTED   = "#6d839a"
MUTED2  = "#93a9c0"
TEXT    = "#bfd0dd"
HEAD    = "#eff6fb"
WHITE   = "#ffffff"
ACC     = "#5ea3d6"     # primary steel blue accent
ACC2    = "#7fc7d9"     # secondary cool cyan accent
ACC3    = "#4b6a88"     # tertiary blueprint accent
DANGER  = "#c96b56"     # controlled terracotta for critical state
WARN    = "#d2a85c"     # muted amber for warning state
OK      = "#7ab7a6"     # industrial teal for healthy state
INFO    = "#8aa5bf"     # cool neutral info tone
FONT_BODY = "'JetBrains Mono', 'IBM Plex Mono', 'Courier New', monospace"
FONT_UI   = "'Space Mono', 'IBM Plex Mono', monospace"

BORDER_R = "2px"
BORDER_W = f"1px solid {GRID}"

# severity → color
SEV_COLOR = {"critical": DANGER, "warning": WARN, "info": INFO}

# ─── CHART LAYOUT HELPER ──────────────────────────────────────────────────────
def _chart_layout(**kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        font={"color": MUTED2, "size": 10, "family": FONT_BODY},
        margin={"l": 44, "r": 20, "t": 20, "b": 36},
        xaxis={"gridcolor": GRID, "zeroline": False, "showgrid": True,
               "linecolor": GRID2, "tickfont": {"size": 9}},
        yaxis={"gridcolor": GRID, "zeroline": False, "showgrid": True,
               "linecolor": GRID2, "tickfont": {"size": 9}},
    )
    for k, v in kwargs.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k].update(v)
        else:
            base[k] = v
    return base

# ─── UI HELPERS ───────────────────────────────────────────────────────────────
def _health_color(score: float) -> str:
    if score >= 80: return OK
    if score >= 55: return WARN
    if score >= 30: return WARN
    return DANGER

def _health_label(score: float) -> str:
    if score >= 80: return "NOMINAL"
    if score >= 55: return "DEGRADED"
    if score >= 30: return "WARNING"
    return "CRITICAL"

def mono_label(text, size="10px", color=MUTED2, bold=False, margin_right="0"):
    return html.Span(text, style={
        "fontFamily": FONT_BODY, "fontSize": size, "color": color,
        "fontWeight": "700" if bold else "400", "marginRight": margin_right,
        "textTransform": "uppercase", "letterSpacing": "0.05em",
    })

def tag(text, color=WHITE, bg="transparent", border_color=None):
    bc = border_color or color
    return html.Span(text, style={
        "fontFamily": FONT_BODY, "fontSize": "9px", "color": color,
        "background": bg or (color + "18"),
        "border": f"1px solid {bc}44",
        "borderRadius": "2px", "padding": "2px 7px",
        "fontWeight": "700", "letterSpacing": "0.07em",
        "textTransform": "uppercase",
    })

def section_hdr(text, color=MUTED2):
    return html.Div(text, style={
        "fontFamily": FONT_BODY, "fontSize": "9px", "fontWeight": "700",
        "color": color, "textTransform": "uppercase",
        "letterSpacing": "0.10em", "marginBottom": "10px",
        "borderBottom": f"1px solid {GRID}", "paddingBottom": "6px",
        "boxShadow": f"inset 0 -1px 0 {color}22",
    })

def kv_row(label, value, value_color=HEAD):
    return html.Div([
        html.Span(label, style={"fontFamily": FONT_BODY, "fontSize": "10px",
                                "color": MUTED, "minWidth": "160px",
                                "display": "inline-block"}),
        html.Span(str(value), style={"fontFamily": FONT_BODY, "fontSize": "10px",
                                     "color": value_color}),
    ], style={"padding": "4px 0", "borderBottom": f"1px solid {GRID}"})

def bool_badge(val):
    on    = bool(val)
    color = OK if on else MUTED
    txt   = "ON" if on else "OFF"
    return tag(txt, color=color)

def sensor_row(label, val):
    is_bool = isinstance(val, bool)
    if is_bool:
        disp = bool_badge(val)
    elif isinstance(val, float):
        disp = html.Span(f"{val:.3f}", style={"color": HEAD, "fontSize": "11px",
                                               "fontFamily": FONT_BODY})
    else:
        disp = html.Span(str(val), style={"color": HEAD, "fontSize": "11px",
                                           "fontFamily": FONT_BODY})
    return html.Div([
        html.Span(label, style={"fontSize": "10px", "color": MUTED,
                                "fontFamily": FONT_BODY, "minWidth": "180px",
                                "display": "inline-block"}),
        html.Span(" ", style={"marginLeft": "8px"}),
        disp,
    ], style={"display": "flex", "alignItems": "center", "padding": "4px 0",
              "borderBottom": f"1px solid {GRID}"})

def section_card(title, color, items, extra=None):
    if not items and extra is None:
        return None
    return html.Div([
        section_hdr(title, color),
        html.Div([sensor_row(k, v) for k, v in (items or [])]),
        extra or html.Div(),
    ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px 14px",
              "flex": "1", "minWidth": "220px",
              "border": BORDER_W, "borderTop": f"2px solid {color}",
              "boxShadow": f"0 12px 30px {BG}55"})

def step_pills(station):
    sm    = sm_state[station]
    steps = STATION_STEPS[station]
    cur   = sm["step_label"]
    hist  = sm["history"]
    pills = []
    for step in steps:
        is_active = (step == cur)
        is_done   = (step in hist and not is_active)
        if is_active:
            bg, border, color, w = ACC2 + "18", ACC2, HEAD, "700"
        elif is_done:
            bg, border, color, w = "transparent", GRID2, MUTED2, "400"
        else:
            bg, border, color, w = "transparent", GRID, MUTED, "400"
        pills.append(html.Div(step, style={
            "padding": "2px 8px", "borderRadius": BORDER_R,
            "fontSize": "9px", "fontWeight": w,
            "background": bg, "border": f"1px solid {border}",
            "color": color, "whiteSpace": "nowrap", "fontFamily": FONT_BODY,
        }))
        if step != steps[-1]:
            pills.append(html.Span("›", style={"color": GRID2, "fontSize": "12px",
                                                "padding": "0 1px"}))
    return pills

def render_station_detail(station):
    import re as _re
    d     = latest.get(station, {})
    name  = MACHINE_NAMES[station]
    icon  = STATION_ICONS[station]
    sm    = sm_state[station]

    state_str = d.get("current_state", "--")
    task_str  = d.get("current_task", "") or "idle"
    sub_str   = d.get("current_sub_task", "") or "--"
    elapsed   = sm["elapsed"]
    failure   = d.get("failure_label", "")

    state_color = (OK     if state_str == "ready"   else
                   WARN   if state_str == "working"  else
                   DANGER if state_str == "error"    else MUTED)

    if not d:
        return html.Div([
            html.Div(f"{icon} {name}", style={"fontSize": "14px", "fontWeight": "700",
                                               "color": HEAD, "fontFamily": FONT_BODY}),
            html.Div("Awaiting data stream...", style={"color": MUTED, "marginTop": "8px",
                                                       "fontSize": "11px", "fontFamily": FONT_BODY}),
        ], style={"background": CARD, "borderRadius": BORDER_R, "padding": "18px",
                  "marginBottom": "12px", "border": BORDER_W})

    sensors, actuators, process_fields = [], [], []
    skip = {"id", "station", "timestamp", "current_stock"}
    for k, v in d.items():
        if k in skip: continue
        if k.startswith("i"):    sensors.append((k, v))
        elif k.startswith(("m", "o")):  actuators.append((k, v))
        elif k.startswith("current") or k == "failure_label": process_fields.append((k, v))
        elif "pos" in k or "target" in k: process_fields.append((k, v))

    # HBW stock grid
    extra_hbw = None
    if station == "HBW_1" and "current_stock" in d:
        stock = d["current_stock"]
        cells = []
        for i in range(9):
            val      = stock.get(str(i), "")
            occupied = bool(val)
            cells.append(html.Div([
                html.Div(f"S{i}", style={"fontSize": "8px", "color": MUTED}),
                html.Div(val if occupied else "·",
                         style={"fontSize": "10px", "fontWeight": "700",
                                "color": WHITE if occupied else MUTED,
                                "marginTop": "2px"}),
            ], style={"background": BG, "borderRadius": BORDER_R, "padding": "6px 3px",
                      "textAlign": "center",
                      "border": f"1px solid {GRID2 if occupied else GRID}",
                      "opacity": "1" if occupied else "0.4"}))
        extra_hbw = html.Div([
            section_hdr("Warehouse Slots", WARN),
            html.Div(cells, style={"display": "grid",
                                   "gridTemplateColumns": "repeat(3, 1fr)", "gap": "4px"}),
        ])

    rows = [s for s in [
        section_card("Sensors",       INFO,  sensors),
        section_card("Actuators",     WARN,  actuators),
        section_card("Process State", WHITE, process_fields,
                     extra=extra_hbw if station == "HBW_1" else None),
    ] if s is not None]

    ts = d.get("timestamp", "")

    with health_lock:
        h_score = health_scores[station]
    with prediction_lock:
        risk = failure_risk[station]
        rul  = predicted_rul[station]

    hcol = _health_color(h_score)

    return html.Div([
        # Header row
        html.Div([
            html.Div([
                html.Span(icon + " ", style={"fontSize": "16px"}),
                html.Span(name, style={"fontSize": "15px", "fontWeight": "700",
                                       "color": HEAD, "fontFamily": FONT_BODY}),
                html.Span(f"  ·  {station}",
                          style={"fontSize": "10px", "color": MUTED,
                                 "fontFamily": FONT_BODY, "marginLeft": "4px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                tag(state_str.upper(), color=state_color),
                html.Span("  "),
                tag(f"H:{h_score:.0f}%", color=hcol),
                html.Span("  "),
                tag(f"RISK:{risk:.0f}%", color=(DANGER if risk > 75 else WARN if risk > 45 else MUTED2)),
            ]),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "10px"}),

        # Step progression
        html.Div([
            html.Div("STEP PROGRESSION", style={"fontSize": "8px", "color": MUTED,
                                                 "fontFamily": FONT_BODY, "marginBottom": "5px",
                                                 "letterSpacing": "0.10em"}),
            html.Div(step_pills(station), style={"display": "flex", "alignItems": "center",
                                                  "flexWrap": "wrap", "gap": "2px"}),
        ], style={"background": BG, "borderRadius": BORDER_R, "padding": "8px 10px",
                  "marginBottom": "8px", "border": BORDER_W}),

        # Task info row
        html.Div([
            html.Div([
                mono_label("Task", color=MUTED),
                html.Div(task_str[:60] + ("…" if len(task_str) > 60 else ""),
                         style={"fontSize": "11px", "color": HEAD, "marginTop": "4px",
                                "fontFamily": FONT_BODY}),
            ], style={"flex": "2", "background": CARD2, "borderRadius": BORDER_R,
                      "padding": "8px 10px", "border": BORDER_W}),
            html.Div([
                mono_label("Sub-task", color=MUTED),
                html.Div(sub_str[:45] + ("…" if len(sub_str) > 45 else ""),
                         style={"fontSize": "10px", "color": TEXT, "marginTop": "4px",
                                "fontFamily": FONT_BODY}),
            ], style={"flex": "2", "background": CARD2, "borderRadius": BORDER_R,
                      "padding": "8px 10px", "border": BORDER_W}),
            html.Div([
                mono_label("Elapsed", color=MUTED),
                html.Div(f"{elapsed:.1f}s", style={"fontSize": "20px", "color": WHITE,
                                                    "marginTop": "4px", "fontFamily": FONT_BODY,
                                                    "fontWeight": "700"}),
            ], style={"flex": "1", "background": CARD2, "borderRadius": BORDER_R,
                      "padding": "8px 10px", "minWidth": "75px", "border": BORDER_W}),
        ], style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "marginBottom": "8px"}),

        # Fault banner
        (html.Div([
            html.Span("[FAULT]: ", style={"fontWeight": "700", "color": DANGER,
                                           "fontFamily": FONT_BODY, "fontSize": "11px"}),
            html.Span(failure, style={"color": DANGER + "cc", "fontFamily": FONT_BODY,
                                      "fontSize": "11px"}),
        ], style={"background": DANGER + "0d", "border": f"1px solid {DANGER}33",
                  "borderRadius": BORDER_R, "padding": "7px 10px",
                  "marginBottom": "8px"})
         if failure else html.Div()),

        html.Div(rows, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        html.Div(f"last update: {ts}",
                 style={"fontSize": "9px", "color": MUTED, "marginTop": "6px",
                        "fontFamily": FONT_BODY}),
    ], style={"background": CARD, "borderRadius": BORDER_R, "padding": "14px 16px",
              "marginBottom": "12px", "border": BORDER_W,
              "borderLeft": f"2px solid {hcol}"})

def make_speedometer_figure(motor_key, raw_val, color):
    rpm       = round(abs(raw_val))
    direction = "fwd" if raw_val > 0 else "rev" if raw_val < 0 else "idle"
    bar_color = DANGER if direction == "rev" else color
    label     = f"{motor_key.replace('_speed','').upper()} · {direction}"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rpm,
        number={"suffix": " RPM", "font": {"size": 13, "color": HEAD, "family": FONT_BODY}},
        title={"text": label, "font": {"size": 10, "color": MUTED2, "family": FONT_BODY}},
        gauge={
            "axis": {"range": [0, MAX_SPEED], "tickwidth": 1, "tickcolor": GRID,
                     "tickfont": {"size": 8, "color": MUTED}, "nticks": 5},
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,            MAX_SPEED * 0.5], "color": "#111"},
                {"range": [MAX_SPEED*0.5, MAX_SPEED*0.8], "color": "#161616"},
                {"range": [MAX_SPEED*0.8, MAX_SPEED],     "color": "#1c1c1c"},
            ],
            "threshold": {"line": {"color": color, "width": 2},
                          "thickness": 0.75, "value": rpm},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin={"l": 18, "r": 18, "t": 36, "b": 8}, height=155,
                      font={"color": MUTED, "family": FONT_BODY})
    return fig

# ─── FLOOR LAYOUT ─────────────────────────────────────────────────────────────
FLOOR_LAYOUT = {
    "MM_1":  {"x": 3.0, "y": 2.0},
    "HBW_1": {"x": 7.0, "y": 2.0},
    "VGR_1": {"x": 5.0, "y": 2.0},
    "SM_1":  {"x": 1.0, "y": 2.0},
    "OV_1":  {"x": 3.0, "y": 5.0},
    "WT_1":  {"x": 5.0, "y": 5.0},
    "EC_1":  {"x": 1.0, "y": 5.0},
}
BELTS = [("SM_1","MM_1"),("MM_1","VGR_1"),("VGR_1","HBW_1"),
         ("VGR_1","OV_1"),("OV_1","WT_1"),("SM_1","EC_1")]
MACHINE_3D_META = {
    "SM_1":  {"z": 0.55, "h": 0.85, "size": 22, "symbol": "square", "shape": "line-ns"},
    "MM_1":  {"z": 0.65, "h": 1.00, "size": 24, "symbol": "square", "shape": "plate"},
    "VGR_1": {"z": 0.90, "h": 1.55, "size": 28, "symbol": "diamond", "shape": "arm"},
    "HBW_1": {"z": 1.20, "h": 2.10, "size": 30, "symbol": "square", "shape": "tower"},
    "OV_1":  {"z": 0.70, "h": 1.15, "size": 24, "symbol": "square", "shape": "plate"},
    "WT_1":  {"z": 0.60, "h": 0.95, "size": 22, "symbol": "square", "shape": "line-ew"},
    "EC_1":  {"z": 0.45, "h": 0.75, "size": 20, "symbol": "circle", "shape": "sensor"},
}

def _make_overview_floor_figure():
    fig = go.Figure()
    for a, b in BELTS:
        ax, ay = FLOOR_LAYOUT[a]["x"], FLOOR_LAYOUT[a]["y"]
        bx, by = FLOOR_LAYOUT[b]["x"], FLOOR_LAYOUT[b]["y"]
        fig.add_trace(go.Scatter(
            x=[ax, bx], y=[ay, by], mode="lines",
            line={"color": GRID2, "width": 2, "dash": "dot"},
            showlegend=False, hoverinfo="skip"
        ))

    with health_lock:
        scores = dict(health_scores)

    for s, pos in FLOOR_LAYOUT.items():
        d = latest.get(s, {})
        state = d.get("current_state", "")
        task = d.get("current_task", "") or "idle"
        sc = scores[s]
        hcol = _health_color(sc)
        name = MACHINE_NAMES[s]
        is_active = state not in ("ready", "") and bool(task and task != "idle")
        ms = 36 if is_active else 26
        op = 1.0 if is_active else 0.5
        hover = (f"<b>{name}</b>  ({s})<br>State: {state}<br>Task: {task[:50]}"
                 f"<br>Health: {sc:.0f}%<br>Elapsed: {d.get('current_task_duration', 0.0):.1f}s")
        if is_active:
            fig.add_trace(go.Scatter(
                x=[pos["x"]], y=[pos["y"]], mode="markers",
                marker={"size": 54, "color": hcol, "opacity": 0.06,
                        "line": {"width": 0}},
                showlegend=False, hoverinfo="skip"
            ))
        fig.add_trace(go.Scatter(
            x=[pos["x"]], y=[pos["y"]], mode="markers+text",
            marker={"size": ms, "color": hcol, "opacity": op,
                    "line": {"width": 2 if is_active else 1, "color": hcol if is_active else GRID2}},
            text=[name[:3]], textposition="middle center",
            textfont={"size": 8, "color": BG, "family": FONT_BODY},
            name=name, showlegend=False,
            hovertemplate=hover + "<extra></extra>",
        ))
        fig.add_annotation(
            x=pos["x"], y=pos["y"] - 0.58, text=name, showarrow=False,
            font={"size": 8, "color": MUTED2, "family": FONT_BODY}, xanchor="center"
        )
        fig.add_annotation(
            x=pos["x"], y=pos["y"] + 0.60, text=f"{sc:.0f}%", showarrow=False,
            font={"size": 9, "color": hcol, "family": FONT_BODY}, xanchor="center"
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"range": [-0.5, 9.5], "showgrid": False, "zeroline": False,
               "showticklabels": False},
        yaxis={"range": [0.5, 7.2], "showgrid": False, "zeroline": False,
               "showticklabels": False, "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest",
    )
    fig.add_shape(
        type="rect", x0=0, y0=0.8, x1=9, y1=6.8,
        line={"color": GRID, "width": 1}, fillcolor="rgba(0,0,0,0)"
    )
    return fig

def _make_factory_3d_figure(flow_phase=0.0):
    with health_lock:
        scores = dict(health_scores)

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=[0, 9, 9, 0], y=[0.8, 0.8, 6.8, 6.8], z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=ACC3, opacity=0.08, hoverinfo="skip", showscale=False, name="Floor"
    ))

    shell_x = [0, 9, 9, 0, 0, None, 0, 9, 9, 0, 0, None, 0, 0, None, 9, 9, None, 9, 9, None, 0, 0]
    shell_y = [0.8, 0.8, 6.8, 6.8, 0.8, None, 0.8, 0.8, 6.8, 6.8, 0.8, None,
               0.8, 0.8, None, 0.8, 0.8, None, 6.8, 6.8, None, 6.8, 6.8]
    shell_z = [0, 0, 0, 0, 0, None, 3.2, 3.2, 3.2, 3.2, 3.2, None,
               0, 3.2, None, 0, 3.2, None, 0, 3.2, None, 0, 3.2]
    fig.add_trace(go.Scatter3d(
        x=shell_x, y=shell_y, z=shell_z, mode="lines",
        line={"color": GRID2, "width": 3},
        opacity=0.35, hoverinfo="skip", name="Factory Envelope"
    ))

    for a, b in BELTS:
        pa = FLOOR_LAYOUT[a]
        pb = FLOOR_LAYOUT[b]
        za = MACHINE_3D_META[a]["z"] + MACHINE_3D_META[a]["h"] * 0.45
        zb = MACHINE_3D_META[b]["z"] + MACHINE_3D_META[b]["h"] * 0.45
        fig.add_trace(go.Scatter3d(
            x=[pa["x"], pb["x"]], y=[pa["y"], pb["y"]], z=[za, zb],
            mode="lines",
            line={"color": ACC2, "width": 5, "dash": "dot"},
            opacity=0.45, hoverinfo="skip", showlegend=False
        ))
        t = flow_phase % 1.0
        mx = pa["x"] + (pb["x"] - pa["x"]) * t
        my = pa["y"] + (pb["y"] - pa["y"]) * t
        mz = za + (zb - za) * t
        fig.add_trace(go.Scatter3d(
            x=[mx], y=[my], z=[mz],
            mode="markers",
            marker={
                "size": 5,
                "color": WARN,
                "symbol": "diamond",
                "opacity": 0.95,
                "line": {"color": HEAD, "width": 1},
            },
            hoverinfo="skip", showlegend=False
        ))

    for s in STATIONS:
        pos = FLOOR_LAYOUT[s]
        meta = MACHINE_3D_META[s]
        d = latest.get(s, {})
        state = d.get("current_state", "")
        task = d.get("current_task", "") or "idle"
        sc = scores[s]
        color = _health_color(sc)
        active = state not in ("ready", "") and task != "idle"
        top_z = meta["z"] + meta["h"]
        label_z = top_z + 0.20

        fig.add_trace(go.Scatter3d(
            x=[pos["x"], pos["x"]], y=[pos["y"], pos["y"]], z=[0, top_z],
            mode="lines",
            line={"color": color + "99", "width": 8 if active else 5},
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[pos["x"]], y=[pos["y"]], z=[top_z],
            mode="markers+text",
            marker={
                "size": meta["size"] + (6 if active else 0),
                "color": color,
                "symbol": meta["symbol"],
                "opacity": 0.96 if active else 0.84,
                "line": {"color": HEAD if active else GRID2, "width": 2},
            },
            text=[s.replace("_1", "")],
            textposition="middle center",
            textfont={"size": 8, "color": BG, "family": FONT_BODY},
            hovertemplate=(f"<b>{MACHINE_NAMES[s]}</b><br>"
                           f"State: {state or '--'}<br>"
                           f"Task: {task[:50]}<br>"
                           f"Health: {sc:.0f}%<extra></extra>"),
            name=MACHINE_NAMES[s],
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[pos["x"]], y=[pos["y"]], z=[label_z],
            mode="text",
            text=[MACHINE_NAMES[s]],
            textfont={"size": 9, "color": HEAD, "family": FONT_BODY},
            hoverinfo="skip", showlegend=False
        ))
        if meta["shape"] == "tower":
            for offset in (-0.18, 0.18):
                fig.add_trace(go.Scatter3d(
                    x=[pos["x"] + offset, pos["x"] + offset],
                    y=[pos["y"] - 0.22, pos["y"] + 0.22],
                    z=[meta["z"], top_z],
                    mode="lines",
                    line={"color": color + "aa", "width": 5},
                    hoverinfo="skip", showlegend=False
                ))
        elif meta["shape"] == "arm":
            fig.add_trace(go.Scatter3d(
                x=[pos["x"] - 0.22, pos["x"], pos["x"] + 0.20],
                y=[pos["y"] - 0.12, pos["y"], pos["y"] + 0.16],
                z=[0.65, top_z - 0.25, top_z + 0.10],
                mode="lines",
                line={"color": HEAD, "width": 7},
                hoverinfo="skip", showlegend=False
            ))
        elif meta["shape"] == "line-ew":
            fig.add_trace(go.Scatter3d(
                x=[pos["x"] - 0.55, pos["x"] + 0.55],
                y=[pos["y"], pos["y"]],
                z=[meta["z"], meta["z"]],
                mode="lines",
                line={"color": color + "bb", "width": 9},
                hoverinfo="skip", showlegend=False
            ))
        elif meta["shape"] == "line-ns":
            fig.add_trace(go.Scatter3d(
                x=[pos["x"], pos["x"]],
                y=[pos["y"] - 0.55, pos["y"] + 0.55],
                z=[meta["z"], meta["z"]],
                mode="lines",
                line={"color": color + "bb", "width": 9},
                hoverinfo="skip", showlegend=False
            ))
        elif meta["shape"] == "sensor":
            fig.add_trace(go.Scatter3d(
                x=[pos["x"], pos["x"]],
                y=[pos["y"], pos["y"]],
                z=[0.20, top_z],
                mode="lines",
                line={"color": color + "aa", "width": 4},
                hoverinfo="skip", showlegend=False
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        font={"color": MUTED2, "size": 10, "family": FONT_BODY},
        scene={
            "bgcolor": BG,
            "aspectmode": "manual",
            "aspectratio": {"x": 1.45, "y": 1.0, "z": 0.55},
            "camera": {"eye": {"x": 1.62, "y": 1.48, "z": 0.88}},
            "xaxis": {"title": "Factory X", "range": [-0.5, 9.5], "gridcolor": GRID,
                      "showbackground": True, "backgroundcolor": CARD, "color": MUTED2},
            "yaxis": {"title": "Factory Y", "range": [0.5, 7.2], "gridcolor": GRID,
                      "showbackground": True, "backgroundcolor": CARD, "color": MUTED2},
            "zaxis": {"title": "Height", "range": [0, 3.4], "gridcolor": GRID,
                      "showbackground": True, "backgroundcolor": CARD, "color": MUTED2},
        },
        annotations=[{
            "text": "Simplified 3D factory system view",
            "xref": "paper", "yref": "paper", "x": 0.01, "y": 1.02,
            "showarrow": False, "font": {"size": 10, "color": MUTED},
        }],
    )
    return fig

# ─── APP INIT ─────────────────────────────────────────────────────────────────
app = Dash(__name__, title="OverTelecomms Engineering · HSG", suppress_callback_exceptions=True)

# CSS injection for slider styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .bw-slider .rc-slider-rail { background: #24364a; }
            .bw-slider .rc-slider-track { background: #5ea3d6; }
            .bw-slider .rc-slider-handle { background: #eff6fb; border-color: #5ea3d6; box-shadow: 0 0 0 4px rgba(94,163,214,0.16); }
            .bw-slider .rc-slider-dot { background: #142235; border-color: #34506d; }
            .bw-slider .rc-slider-dot-active { background: #7fc7d9; border-color: #7fc7d9; }
            .bw-slider .rc-slider-mark-text { color: #6d839a; font-family: 'JetBrains Mono', monospace; font-size: 9px; }
            ::-webkit-scrollbar { width: 4px; height: 4px; }
            ::-webkit-scrollbar-track { background: #101b2a; }
            ::-webkit-scrollbar-thumb { background: #34506d; }
            body {
                background:
                    radial-gradient(circle at top left, rgba(94,163,214,0.12), transparent 28%),
                    radial-gradient(circle at top right, rgba(122,183,166,0.08), transparent 24%),
                    linear-gradient(180deg, #0a1420 0%, #08111d 42%, #060d16 100%) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────
app.layout = html.Div(style={
    "background": BG, "minHeight": "100vh", "fontFamily": FONT_BODY,
    "color": TEXT, "padding": "16px 20px",
    "backgroundImage": f"radial-gradient(circle at top left, {ACC}1c 0, transparent 24%), "
                       f"radial-gradient(circle at 85% 0%, {OK}14 0, transparent 22%)",
}, children=[

    # ── HEADER ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div([
                html.Span("◼ ", style={"color": ACC2, "fontSize": "18px"}),
                html.Span("SMART FACTORY", style={"fontSize": "17px", "fontWeight": "700",
                                                   "color": HEAD, "letterSpacing": "0.12em"}),
                html.Span(" / PREDICTIVE MONITOR",
                          style={"fontSize": "11px", "color": INFO, "marginLeft": "8px",
                                 "letterSpacing": "0.06em"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div("FISCHERTECHNIK · Pty Ltd. © 2024",
                     style={"fontSize": "9px", "color": MUTED2, "marginTop": "3px",
                            "letterSpacing": "0.12em"}),
        ]),
        html.Div([
            html.Button("[PLAY]", id="btn-play",
                        style={"background": ACC, "color": HEAD, "border": f"1px solid {ACC2}",
                               "borderRadius": BORDER_R, "padding": "7px 16px",
                               "cursor": "pointer", "fontSize": "11px", "fontWeight": "700",
                               "marginRight": "8px", "fontFamily": FONT_BODY,
                               "letterSpacing": "0.08em",
                               "boxShadow": f"0 8px 20px {ACC}26"}),
            html.Button("[PAUSE]", id="btn-pause",
                        style={"background": "transparent", "color": TEXT,
                               "border": f"1px solid {GRID2}",
                               "borderRadius": BORDER_R, "padding": "7px 14px",
                               "cursor": "pointer", "fontSize": "11px", "marginRight": "14px",
                               "fontFamily": FONT_BODY, "letterSpacing": "0.08em"}),
            html.Div([
                html.Span("SPEED ", style={"fontSize": "9px", "color": MUTED,
                                            "marginRight": "8px", "letterSpacing": "0.10em"}),
                dcc.Slider(id="speed-slider", min=1, max=50, step=1, value=5,
                           marks={1:"1×", 10:"10×", 25:"25×", 50:"50×"},
                           tooltip={"placement": "top"},
                           className="bw-slider"),
            ], style={"display": "flex", "alignItems": "center", "minWidth": "200px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "marginBottom": "12px", "borderBottom": f"1px solid {GRID}",
              "paddingBottom": "12px", "background": CARD + "cc",
              "border": BORDER_W, "borderRadius": BORDER_R,
              "padding": "14px 16px", "boxShadow": f"0 18px 40px {BG}44"}),

    # ── PROGRESS BAR ────────────────────────────────────────────────────────
    html.Div([
        html.Div(id="progress-bar-fill",
                 style={"height": "2px", "background": ACC2, "width": "0%",
                        "transition": "width 0.4s ease"}),
    ], style={"background": GRID, "height": "2px", "marginBottom": "4px"}),
    html.Div(id="progress-label",
             style={"fontSize": "9px", "color": MUTED, "marginBottom": "12px",
                    "textAlign": "right", "letterSpacing": "0.06em"}),

    # ── GLOBAL ALERT BANNER ─────────────────────────────────────────────────
    html.Div(id="global-alert-banner", style={"marginBottom": "12px"}),

    # ── KPI STRIP ───────────────────────────────────────────────────────────
    html.Div(id="kpi-row",
             style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "marginBottom": "14px"}),

    # ── TABS ────────────────────────────────────────────────────────────────
    dcc.Tabs(id="main-tabs", value="overview",
             colors={"border": GRID, "primary": ACC2, "background": BG},
             style={"marginBottom": "0", "background": "transparent"},
             children=[
                 dcc.Tab(label="OVERVIEW",     value="overview",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="PRODUCTION",   value="production",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="LOGISTICS",    value="logistics",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="SYSTEM FLOW",  value="system_flow",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="TELEMETRY",    value="telemetry",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="EC SIGNAL",    value="ec_signal",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="EVENT LOG",    value="event_log",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="HEALTH",       value="health",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="PREDICTIVE MAINTENANCE", value="predictive",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
                 dcc.Tab(label="DRILL-DOWN",   value="drill_down",
                         style={"color": MUTED, "background": BG, "border": "none",
                                "padding": "8px 14px", "fontSize": "10px",
                                "fontFamily": FONT_BODY, "letterSpacing": "0.08em"},
                         selected_style={"color": HEAD, "background": CARD2,
                                         "border": f"1px solid {GRID2}",
                                         "borderBottom": f"1px solid {CARD2}",
                                         "borderTop": f"2px solid {ACC}",
                                         "fontWeight": "700"}),
             ]),

    html.Div(id="tab-content",
             style={"background": CARD, "borderRadius": "0 0 2px 2px",
                    "padding": "16px", "minHeight": "400px",
                    "border": BORDER_W, "borderTop": "none",
                    "boxShadow": f"0 20px 44px {BG}3f"}),

    dcc.Interval(id="interval", interval=600, n_intervals=0),

])

# ─── PROGRESS ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("progress-bar-fill", "style"),
    Output("progress-label", "children"),
    Input("interval", "n_intervals"),
)
def update_progress(_):
    total   = playback["total_lines"]
    current = playback["current_line"]
    pct     = (current / total * 100) if total else 0
    status  = ("PLAYING" if (playback["running"] and not playback["paused"])
               else ("PAUSED" if playback["paused"] else "STOPPED"))
    label   = f"{status}  ·  {current:,} / {total:,}  ·  {pct:.1f}%  ·  {playback['speed']:.0f}×"
    return {"height": "2px", "background": WHITE, "width": f"{pct:.1f}%",
            "transition": "width 0.4s ease"}, label

# ─── GLOBAL ALERT BANNER ──────────────────────────────────────────────────────
@app.callback(
    Output("global-alert-banner", "children"),
    Input("interval", "n_intervals"),
)
def update_alert_banner(_):
    with alerts_lock:
        recent = list(active_alerts)[-6:]
    if not recent:
        return html.Div()
    rows = []
    for a in reversed(recent):
        col = SEV_COLOR.get(a["severity"], MUTED2)
        rows.append(html.Div([
            html.Span(f"[{a['severity'].upper()}]",
                      style={"color": col, "fontWeight": "700", "marginRight": "8px",
                             "fontSize": "9px", "letterSpacing": "0.08em"}),
            html.Span(MACHINE_NAMES.get(a["station"], a["station"]),
                      style={"color": col, "marginRight": "8px", "fontSize": "10px"}),
            html.Span(a["msg"], style={"color": TEXT, "fontSize": "10px"}),
            html.Span(f" · {str(a['ts'])[-12:]}",
                      style={"color": MUTED, "fontSize": "9px", "marginLeft": "8px"}),
        ], style={"padding": "4px 10px", "borderBottom": f"1px solid {GRID}",
                  "display": "flex", "alignItems": "center"}))
    return html.Div([
        html.Div("LIVE ALERTS", style={"fontSize": "8px", "fontWeight": "700",
                                        "color": MUTED, "letterSpacing": "0.14em",
                                        "padding": "5px 10px 3px",
                                        "borderBottom": f"1px solid {GRID}"}),
        html.Div(rows),
    ], style={"background": CARD, "border": f"1px solid {GRID2}",
              "borderLeft": f"2px solid {DANGER}", "borderRadius": BORDER_R,
              "fontFamily": FONT_BODY})

# ─── KPI STRIP ────────────────────────────────────────────────────────────────
@app.callback(Output("kpi-row", "children"), Input("interval", "n_intervals"))
def update_kpi(_):
    with health_lock:
        scores = dict(health_scores)
    with prediction_lock:
        risks = dict(failure_risk)
    cards = []
    for s in STATIONS:
        d      = latest.get(s, {})
        state  = d.get("current_state", "--")
        sc     = scores[s]
        rk     = risks[s]
        hcol   = _health_color(sc)
        scol   = (OK if state == "ready" else WARN if state == "working"
                  else DANGER if state == "error" else MUTED)
        kpi    = kpi_store[s]
        cards.append(html.Div([
            html.Div([
                html.Span(STATION_ICONS[s] + " ", style={"fontSize": "12px"}),
                html.Span(MACHINE_NAMES[s],
                          style={"fontSize": "10px", "fontWeight": "700",
                                 "color": HEAD, "letterSpacing": "0.04em"}),
            ], style={"marginBottom": "4px"}),
            html.Div(s, style={"fontSize": "8px", "color": MUTED, "marginBottom": "6px",
                               "letterSpacing": "0.08em"}),
            # Health bar
            html.Div([
                html.Div(style={"height": "3px", "background": hcol,
                                "width": f"{sc:.0f}%", "transition": "width 0.5s"}),
            ], style={"background": GRID, "height": "3px", "borderRadius": "1px",
                      "marginBottom": "5px"}),
            html.Div([
                html.Span(f"{sc:.0f}%", style={"color": hcol, "fontWeight": "700",
                                                "fontSize": "14px", "marginRight": "8px"}),
                html.Span(f"RISK {rk:.0f}%",
                          style={"color": DANGER if rk > 75 else WARN if rk > 45 else MUTED2,
                                 "fontSize": "9px", "letterSpacing": "0.05em"}),
            ], style={"display": "flex", "alignItems": "baseline"}),
            html.Div([
                html.Span(state.upper(), style={"color": scol, "fontSize": "8px",
                                                "letterSpacing": "0.08em", "marginRight": "6px"}),
                html.Span(f"F:{kpi['faults_total']}",
                          style={"color": DANGER if kpi["faults_total"] else MUTED,
                                 "fontSize": "8px", "marginRight": "4px"}),
                html.Span(f"A:{kpi['motor_anomalies']}",
                          style={"color": WARN if kpi["motor_anomalies"] else MUTED,
                                 "fontSize": "8px"}),
            ], style={"marginTop": "3px"}),
        ], style={"background": CARD, "borderRadius": BORDER_R,
                  "padding": "9px 12px", "minWidth": "130px",
                  "border": BORDER_W, "borderTop": f"2px solid {hcol}"}))
    return cards

# ─── TAB ROUTER ───────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("btn-play",  "n_clicks"),
    Input("btn-pause", "n_clicks"),
    State("speed-slider", "value"),
)
def render_tab(tab, play_clicks, pause_clicks, speed):
    if callback_context.triggered_id == "btn-play":
        playback["speed"]  = float(speed)
        playback["paused"] = False
        if not playback["running"] and playback.get("log_file"):
            playback["running"] = True
            t = threading.Thread(target=run_playback, args=(playback["log_file"],), daemon=True)
            t.start()
    elif callback_context.triggered_id == "btn-pause":
        playback["paused"] = True

    if tab == "overview":   return build_overview()
    if tab == "production": return build_production()
    if tab == "logistics":  return build_logistics()
    if tab == "system_flow": return build_system_flow()
    if tab == "telemetry":  return build_telemetry()
    if tab == "ec_signal":  return build_ec_signal()
    if tab == "event_log":  return build_event_log()
    if tab == "health":     return build_health()
    if tab == "predictive": return build_predictive()
    if tab == "drill_down": return build_drill_down()
    return html.Div("Unknown tab")

# ─── TAB: OVERVIEW ────────────────────────────────────────────────────────────
def build_overview():
    fig = go.Figure()
    for a, b in BELTS:
        ax, ay = FLOOR_LAYOUT[a]["x"], FLOOR_LAYOUT[a]["y"]
        bx, by = FLOOR_LAYOUT[b]["x"], FLOOR_LAYOUT[b]["y"]
        fig.add_trace(go.Scatter(x=[ax, bx], y=[ay, by], mode="lines",
                                  line={"color": GRID2, "width": 2, "dash": "dot"},
                                  showlegend=False, hoverinfo="skip"))

    with health_lock:
        scores = dict(health_scores)

    for s, pos in FLOOR_LAYOUT.items():
        d       = latest.get(s, {})
        state   = d.get("current_state", "")
        task    = d.get("current_task", "") or "idle"
        sub     = d.get("current_sub_task", "") or ""
        sc      = scores[s]
        hcol    = _health_color(sc)
        name    = MACHINE_NAMES[s]
        is_active = state not in ("ready", "") and bool(task and task != "idle")
        ms      = 36 if is_active else 26
        op      = 1.0 if is_active else 0.5
        hover   = (f"<b>{name}</b>  ({s})<br>State: {state}<br>Task: {task[:50]}"
                   f"<br>Health: {sc:.0f}%<br>Elapsed: {d.get('current_task_duration', 0.0):.1f}s")
        if is_active:
            fig.add_trace(go.Scatter(x=[pos["x"]], y=[pos["y"]], mode="markers",
                                      marker={"size": 54, "color": hcol, "opacity": 0.06,
                                              "line": {"width": 0}},
                                      showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[pos["x"]], y=[pos["y"]], mode="markers+text",
            marker={"size": ms, "color": hcol, "opacity": op,
                    "line": {"width": 2 if is_active else 1, "color": hcol if is_active else GRID2}},
            text=[name[:3]], textposition="middle center",
            textfont={"size": 8, "color": BG, "family": FONT_BODY},
            name=name, showlegend=False,
            hovertemplate=hover + "<extra></extra>",
        ))
        fig.add_annotation(x=pos["x"], y=pos["y"] - 0.58, text=name,
                           showarrow=False,
                           font={"size": 8, "color": MUTED2, "family": FONT_BODY},
                           xanchor="center")
        fig.add_annotation(x=pos["x"], y=pos["y"] + 0.60,
                           text=f"{sc:.0f}%",
                           showarrow=False,
                           font={"size": 9, "color": hcol, "family": FONT_BODY},
                           xanchor="center")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"range": [-0.5, 9.5], "showgrid": False, "zeroline": False,
               "showticklabels": False},
        yaxis={"range": [0.5, 7.2], "showgrid": False, "zeroline": False,
               "showticklabels": False, "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest",
    )
    fig.add_shape(type="rect", x0=0, y0=0.8, x1=9, y1=6.8,
                  line={"color": GRID, "width": 1}, fillcolor="rgba(0,0,0,0)")

    # Recent anomalies
    recent_failures = list(failure_log)[-8:]
    fail_rows = []
    for e in reversed(recent_failures):
        fail_rows.append(html.Div([
            html.Span(e["ts"][-12:],
                      style={"color": MUTED, "marginRight": "8px", "fontSize": "9px"}),
            html.Span(MACHINE_NAMES.get(e["station"], e["station"]),
                      style={"color": WHITE, "fontWeight": "700", "marginRight": "8px",
                             "fontSize": "10px"}),
            html.Span(e["label"], style={"color": DANGER, "fontSize": "10px"}),
        ], style={"padding": "4px 0", "borderBottom": f"1px solid {GRID}"}))
    if not fail_rows:
        fail_rows = [html.Div("No anomalies detected.",
                              style={"color": MUTED, "fontSize": "10px", "padding": "6px 0"})]

    # SM overview
    sm_rows = []
    for s in STATIONS:
        sm  = sm_state[s]
        sc  = scores[s]
        hcol = _health_color(sc)
        task = (sm["task"] or "idle")[:40]
        sm_rows.append(html.Div([
            html.Div(style={"width": "6px", "height": "6px", "borderRadius": "50%",
                             "background": hcol, "flexShrink": "0", "marginTop": "4px"}),
            html.Div([
                html.Span(MACHINE_NAMES[s], style={"fontSize": "11px", "fontWeight": "700",
                                                    "color": HEAD}),
                html.Span(f"  {s}", style={"fontSize": "9px", "color": MUTED}),
                html.Div(task, style={"fontSize": "9px", "color": MUTED2,
                                      "marginTop": "1px"}),
            ], style={"marginLeft": "10px", "minWidth": "180px"}),
            html.Div(step_pills(s), style={"display": "flex", "alignItems": "center",
                                            "flexWrap": "wrap", "gap": "2px", "flex": "1"}),
        ], style={"display": "flex", "alignItems": "flex-start", "padding": "7px 0",
                  "gap": "8px",
                  "borderBottom": f"1px solid {GRID}" if s != STATIONS[-1] else "none"}))

    return html.Div([
        html.Div([
            html.Div([
                section_hdr("Factory Floor", WHITE),
                dcc.Graph(figure=fig, style={"height": "340px"},
                          config={"displayModeBar": False}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "2", "minWidth": "340px", "border": BORDER_W}),
            html.Div([
                section_hdr("Recent Anomalies", DANGER),
                html.Div(fail_rows, style={"maxHeight": "150px", "overflowY": "auto"}),
                html.Div(style={"height": "1px", "background": GRID, "margin": "14px 0"}),
                section_hdr("Machine Status", MUTED2),
                html.Div([
                    html.Div([
                        html.Span(MACHINE_NAMES[s],
                                  style={"fontSize": "10px", "color": HEAD, "fontWeight": "600"}),
                        html.Span(
                            latest.get(s, {}).get("current_state", "--").upper(),
                            style={"fontSize": "9px", "letterSpacing": "0.06em",
                                   "color": OK    if latest.get(s,{}).get("current_state") == "ready"
                                            else WARN  if latest.get(s,{}).get("current_state") == "working"
                                            else DANGER if latest.get(s,{}).get("current_state") == "error"
                                            else MUTED}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "padding": "4px 0"})
                    for s in STATIONS
                ]),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "1", "minWidth": "220px", "border": BORDER_W}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "14px"}),

        html.Div([
            section_hdr("Step Progression — All Machines", MUTED2),
            html.Div(sm_rows),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "14px",
                  "border": BORDER_W, "marginBottom": "14px"}),

    ])

# ─── TAB: PRODUCTION ──────────────────────────────────────────────────────────
def build_system_flow():
    chips = []
    with health_lock:
        scores = dict(health_scores)
    for s in STATIONS:
        d = latest.get(s, {})
        state = d.get("current_state", "--").upper()
        task = (d.get("current_task", "") or "idle")[:28]
        col = _health_color(scores[s])
        chips.append(html.Div([
            html.Div(MACHINE_NAMES[s], style={"fontSize": "10px", "fontWeight": "700", "color": HEAD}),
            html.Div(state, style={"fontSize": "8px", "letterSpacing": "0.08em", "color": col, "marginTop": "2px"}),
            html.Div(task, style={"fontSize": "8px", "color": MUTED2, "marginTop": "4px"}),
        ], style={"background": CARD, "border": BORDER_W, "borderTop": f"2px solid {col}",
                  "borderRadius": BORDER_R, "padding": "10px 12px", "minWidth": "140px", "flex": "1"}))

    return html.Div([
        section_hdr("System Flow â€” 3D Factory Coordination View", ACC2),
        html.Div([
            dcc.Graph(id="system-flow-3d", figure=_make_factory_3d_figure(0.0),
                      style={"height": "520px"},
                      config={"displayModeBar": False}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "border": BORDER_W, "marginBottom": "12px"}),
        html.Div(chips, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
    ])

@app.callback(
    Output("system-flow-3d", "figure"),
    Input("interval", "n_intervals"),
    Input("main-tabs", "value"),
)
def update_system_flow_3d(n_intervals, tab):
    if tab != "system_flow":
        raise PreventUpdate
    return _make_factory_3d_figure((n_intervals % 24) / 24.0)

def build_production():
    return html.Div([
        section_hdr("Production Machines — Milling · Oven · Sorting", MUTED2),
        html.Div([render_station_detail(s) for s in PRODUCTION_STATIONS]),
    ])

# ─── TAB: LOGISTICS ───────────────────────────────────────────────────────────
def build_logistics():
    return html.Div([
        section_hdr("Logistics & Handling — Robot Arm · Warehouse · Transfer", MUTED2),
        html.Div([render_station_detail(s) for s in LOGISTICS_STATIONS]),
        html.Div([
            section_hdr("Movement Paths — Robot Arm & Warehouse Carrier", MUTED2),
            dcc.Graph(id="movement-map-logistics", style={"height": "280px"},
                      config={"displayModeBar": False}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "marginTop": "4px", "border": BORDER_W}),
    ])

@app.callback(
    Output("movement-map-logistics", "figure"),
    Input("interval", "n_intervals"),
    Input("main-tabs", "value"),
)
def update_movement_logistics(_, tab):
    if tab != "logistics":
        return go.Figure()
    fig = go.Figure()
    hbw = pos_history["HBW_1"]
    hx, hy = list(hbw["x"]), list(hbw["y"])
    if len(hx) > 1:
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines",
                                  line={"color": MUTED2, "width": 1.2}, opacity=0.5,
                                  name="Warehouse Carrier", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[hx[-1]], y=[hy[-1]], mode="markers",
                                  marker={"size": 10, "color": WHITE, "symbol": "square",
                                          "line": {"width": 1, "color": BG}},
                                  name="Carrier now"))
    vgr = pos_history["VGR_1"]
    vx, vy = list(vgr["x"]), list(vgr["y"])
    if len(vx) > 1:
        fig.add_trace(go.Scatter(x=vx, y=[y + 180 for y in vy], mode="lines",
                                  line={"color": ACC2, "width": 1.2}, opacity=0.5,
                                  name="Robot Arm", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[vx[-1]], y=[vy[-1]+180], mode="markers",
                                  marker={"size": 10, "color": WHITE, "symbol": "circle",
                                          "line": {"width": 1, "color": BG}},
                                  name="Robot now"))
    fig.update_layout(**_chart_layout(
        legend={"orientation": "h", "y": -0.15,
                "font": {"color": MUTED2, "size": 10, "family": FONT_BODY},
                "bgcolor": "rgba(0,0,0,0)"},
        xaxis={"title": "X (mm)", "gridcolor": GRID, "range": [-20, 500]},
        yaxis={"title": "Y (mm)", "gridcolor": GRID, "range": [-50, 340]},
    ))
    return fig

# ─── TAB: TELEMETRY ───────────────────────────────────────────────────────────
def build_telemetry():
    return html.Div([
        section_hdr("Telemetry — Motor Speeds & Positions", MUTED2),
        html.Div([
            html.Span("MACHINE:", style={"fontSize": "9px", "color": MUTED,
                                          "marginRight": "10px", "letterSpacing": "0.08em"}),
            dcc.Dropdown(id="telem-station-select",
                         options=[{"label": MACHINE_NAMES[s], "value": s} for s in STATIONS],
                         value="MM_1", clearable=False,
                         style={"width": "260px", "background": CARD2, "color": HEAD,
                                "border": f"1px solid {GRID2}", "borderRadius": BORDER_R,
                                "fontFamily": FONT_BODY, "fontSize": "11px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),

        html.Div([
            html.Div([
                section_hdr("Motor Speedometers", MUTED2),
                html.Div([
                    dcc.Graph(id="telem-gauge-m1", style={"height": "155px"},
                              config={"displayModeBar": False}),
                    dcc.Graph(id="telem-gauge-m2", style={"height": "155px"},
                              config={"displayModeBar": False}),
                    dcc.Graph(id="telem-gauge-m3", style={"height": "155px"},
                              config={"displayModeBar": False}),
                    dcc.Graph(id="telem-gauge-m4", style={"height": "155px"},
                              config={"displayModeBar": False}),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(145px, 1fr))",
                          "gap": "4px"}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "1", "minWidth": "300px", "border": BORDER_W}),
            html.Div([
                html.Div(id="telem-pos-title",
                         style={"fontSize": "9px", "fontWeight": "700", "color": MUTED2,
                                "textTransform": "uppercase", "letterSpacing": "0.08em",
                                "marginBottom": "8px", "borderBottom": f"1px solid {GRID}",
                                "paddingBottom": "6px"}),
                dcc.Graph(id="telem-pos-chart", style={"height": "215px"},
                          config={"displayModeBar": False}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "1", "minWidth": "300px", "border": BORDER_W}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

        html.Div([
            html.Div([
                html.Div(id="telem-vgr-3d-title",
                         style={"fontSize": "9px", "fontWeight": "700", "color": MUTED2,
                                "textTransform": "uppercase", "letterSpacing": "0.08em",
                                "marginBottom": "8px", "borderBottom": f"1px solid {GRID}",
                                "paddingBottom": "6px"}),
                dcc.Graph(id="telem-vgr-3d", style={"height": "320px"},
                          config={"displayModeBar": False}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "1", "minWidth": "320px", "border": BORDER_W}),
            html.Div([
                section_hdr("Motor Speed History", MUTED2),
                dcc.Graph(id="telem-speed-history", style={"height": "320px"},
                          config={"displayModeBar": False}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "1", "minWidth": "320px", "border": BORDER_W}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ])

@app.callback(
    Output("telem-gauge-m1", "figure"),
    Output("telem-gauge-m2", "figure"),
    Output("telem-gauge-m3", "figure"),
    Output("telem-gauge-m4", "figure"),
    Input("interval", "n_intervals"),
    Input("telem-station-select", "value"),
    Input("main-tabs", "value"),
)
def update_motor_gauges(_1, station, tab):
    if tab != "telemetry":
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    shades = [WHITE, ACC2, ACC3, MUTED2]
    def _gauge(mk, color):
        mh   = motor_history.get(station, {})
        hist = mh.get(mk, [])
        raw  = hist[-1][1] if hist else 0
        return make_speedometer_figure(mk, raw, color)
    return (_gauge("m1_speed", shades[0]), _gauge("m2_speed", shades[1]),
            _gauge("m3_speed", shades[2]), _gauge("m4_speed", shades[3]))

@app.callback(
    Output("telem-pos-chart", "figure"),
    Output("telem-pos-title", "children"),
    Input("interval", "n_intervals"),
    Input("telem-station-select", "value"),
    Input("main-tabs", "value"),
)
def update_telem_pos(_, station, tab):
    name  = MACHINE_NAMES.get(station, station)
    title = f"Position Tracking — {name}"
    if tab != "telemetry":
        return go.Figure(), title
    fig = go.Figure()
    if station == "VGR_1":
        ph = pos_history["VGR_1"]
        ts = list(ph["ts"])
        for axis, color in [("x", WHITE), ("y", ACC2), ("z", ACC3)]:
            vals = list(ph[axis])
            if vals:
                fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name=axis.upper(),
                                          line={"color": color, "width": 1.5}))
    elif station == "HBW_1":
        ph = pos_history["HBW_1"]
        ts = list(ph["ts"])
        for axis, color in [("x", WHITE), ("y", ACC2)]:
            vals = list(ph[axis])
            if vals:
                fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name=axis.upper(),
                                          line={"color": color, "width": 1.5}))
    else:
        fig.add_annotation(text=f"No position data for {name}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"color": MUTED, "size": 12,
                                                  "family": FONT_BODY})
    fig.update_layout(**_chart_layout(
        legend={"orientation": "h", "y": 1.08, "bgcolor": "rgba(0,0,0,0)",
                "font": {"color": MUTED2, "size": 9, "family": FONT_BODY}},
        yaxis={"gridcolor": GRID, "title": "Position (mm)"},
    ))
    return fig, title

@app.callback(
    Output("telem-vgr-3d", "figure"),
    Output("telem-vgr-3d-title", "children"),
    Input("interval", "n_intervals"),
    Input("telem-station-select", "value"),
    Input("main-tabs", "value"),
)
def update_vgr_3d(_, station, tab):
    title = "3D Robot Arm View"
    if tab != "telemetry":
        return go.Figure(), title

    fig = go.Figure()
    if station != "VGR_1":
        fig.add_annotation(
            text=f"3D robot view is available for {MACHINE_NAMES['VGR_1']}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font={"color": MUTED, "size": 12, "family": FONT_BODY}
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
                "bgcolor": BG,
            },
        )
        return fig, title + " - VGR only"

    ph = pos_history["VGR_1"]
    xs = list(ph["x"])
    ys = list(ph["y"])
    zs = list(ph["z"])
    if not xs or not ys or not zs:
        fig.add_annotation(
            text="Press PLAY to start the robot arm visualization",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font={"color": MUTED, "size": 12, "family": FONT_BODY}
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        return fig, title

    x, y, z = float(xs[-1]), float(ys[-1]), float(zs[-1])

    # Inferred linkage from the available end-effector coordinates so the
    # robot reads as a physical arm instead of a single point cloud.
    base = (0.0, 0.0, 0.0)
    shoulder = (0.0, 0.0, 110.0)
    elbow = (
        max(min(x * 0.48, 240.0), -40.0),
        max(min(y * 0.48, 240.0), -40.0),
        max(min(z + 55.0, 250.0), 130.0),
    )
    wrist = (x, y, z)

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name="Toolpath",
        line={"color": ACC2, "width": 5},
        opacity=0.34,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[base[0], shoulder[0], elbow[0], wrist[0]],
        y=[base[1], shoulder[1], elbow[1], wrist[1]],
        z=[base[2], shoulder[2], elbow[2], wrist[2]],
        mode="lines+markers",
        name="Arm",
        line={"color": HEAD, "width": 8},
        marker={
            "size": [7, 6, 6, 8],
            "color": [ACC3, ACC, ACC2, WARN],
            "line": {"color": BG, "width": 1},
        },
        hovertemplate="X %{x:.1f}<br>Y %{y:.1f}<br>Z %{z:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[wrist[0]],
        y=[wrist[1]],
        z=[wrist[2]],
        mode="markers",
        name="End Effector",
        marker={
            "size": 10,
            "color": WARN,
            "symbol": "diamond",
            "line": {"color": HEAD, "width": 1},
        },
        hovertemplate="End Effector<br>X %{x:.1f}<br>Y %{y:.1f}<br>Z %{z:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 420, 420, 0, 0, None, 0, 420, 420, 0, 0, None, 0, 0, None, 420, 420, None, 420, 420, None, 0, 0],
        y=[0, 0, 240, 240, 0, None, 0, 0, 240, 240, 0, None, 0, 0, None, 0, 0, None, 240, 240, None, 240, 240],
        z=[0, 0, 0, 0, 0, None, 280, 280, 280, 280, 280, None, 0, 280, None, 0, 280, None, 0, 280, None, 0, 280],
        mode="lines",
        name="Work Envelope",
        line={"color": GRID2, "width": 2, "dash": "dot"},
        opacity=0.22,
        hoverinfo="skip",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        font={"color": MUTED2, "size": 10, "family": FONT_BODY},
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 0.01,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": MUTED2, "size": 9, "family": FONT_BODY},
        },
        scene={
            "bgcolor": BG,
            "aspectmode": "manual",
            "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.9},
            "camera": {"eye": {"x": 1.65, "y": 1.45, "z": 1.15}},
            "xaxis": {
                "title": "X (mm)",
                "range": [-20, 440],
                "gridcolor": GRID,
                "showbackground": True,
                "backgroundcolor": CARD,
                "zerolinecolor": GRID2,
                "color": MUTED2,
            },
            "yaxis": {
                "title": "Y (mm)",
                "range": [-20, 260],
                "gridcolor": GRID,
                "showbackground": True,
                "backgroundcolor": CARD,
                "zerolinecolor": GRID2,
                "color": MUTED2,
            },
            "zaxis": {
                "title": "Z (mm)",
                "range": [0, 300],
                "gridcolor": GRID,
                "showbackground": True,
                "backgroundcolor": CARD,
                "zerolinecolor": GRID2,
                "color": MUTED2,
            },
        },
    )
    return fig, f"{title} - {MACHINE_NAMES['VGR_1']}"

@app.callback(
    Output("telem-speed-history", "figure"),
    Input("interval", "n_intervals"),
    Input("telem-station-select", "value"),
    Input("main-tabs", "value"),
)
def update_speed_history(_, station, tab):
    if tab != "telemetry":
        return go.Figure()
    fig    = go.Figure()
    colors = {"m1_speed": WHITE, "m2_speed": ACC2, "m3_speed": ACC3, "m4_speed": MUTED2}
    mh     = motor_history.get(station, {})
    for mk, color in colors.items():
        hist = list(mh.get(mk, []))
        if hist:
            fig.add_trace(go.Scatter(
                x=[h[0] for h in hist], y=[h[1] for h in hist],
                mode="lines", name=mk.replace("_speed", "").upper(),
                line={"color": color, "width": 1.5},
            ))
    if not any(mh.get(mk) for mk in colors):
        fig.add_annotation(text="No motor data yet", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font={"color": MUTED, "size": 12, "family": FONT_BODY})
    fig.update_layout(**_chart_layout(
        legend={"orientation": "h", "y": 1.08, "bgcolor": "rgba(0,0,0,0)",
                "font": {"color": MUTED2, "size": 9, "family": FONT_BODY}},
        yaxis={"gridcolor": GRID, "title": "Speed (RPM)", "zeroline": True,
               "zerolinecolor": GRID2},
    ))
    return fig

# ─── TAB: EC SIGNAL ───────────────────────────────────────────────────────────
def build_ec_signal():
    return html.Div([
        section_hdr("EC Signal Monitor — Env. Controller (EC_1)", MUTED2),
        html.Div([
            html.Div(id="ec-kpi-mean",     style={"flex":"1","minWidth":"110px"}),
            html.Div(id="ec-kpi-rms",      style={"flex":"1","minWidth":"110px"}),
            html.Div(id="ec-kpi-variance", style={"flex":"1","minWidth":"110px"}),
            html.Div(id="ec-kpi-max",      style={"flex":"1","minWidth":"110px"}),
            html.Div(id="ec-kpi-ber",      style={"flex":"1","minWidth":"110px"}),
            html.Div(id="ec-kpi-state",    style={"flex":"1","minWidth":"110px"}),
        ], style={"display":"flex","gap":"8px","flexWrap":"wrap","marginBottom":"12px"}),

        html.Div([
            html.Div([
                section_hdr("BER Trend", MUTED2),
                dcc.Graph(id="ec-ber-chart", style={"height":"180px"},
                          config={"displayModeBar":False}),
            ], style={"background":CARD2,"borderRadius":BORDER_R,"padding":"12px",
                      "flex":"1","minWidth":"280px","border":BORDER_W}),
            html.Div([
                section_hdr("State Timeline  (0=OK · 1=Anomaly)", MUTED2),
                dcc.Graph(id="ec-state-chart", style={"height":"180px"},
                          config={"displayModeBar":False}),
            ], style={"background":CARD2,"borderRadius":BORDER_R,"padding":"12px",
                      "flex":"1","minWidth":"280px","border":BORDER_W}),
        ], style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"12px"}),

        html.Div([
            section_hdr("Latest EC Record", MUTED2),
            html.Pre(id="ec-raw-record",
                     style={"fontSize":"10px","color":MUTED2,"fontFamily":FONT_BODY,
                            "margin":0,"whiteSpace":"pre-wrap","wordBreak":"break-all",
                            "maxHeight":"110px","overflowY":"auto"}),
        ], style={"background":CARD2,"borderRadius":BORDER_R,"padding":"12px","border":BORDER_W}),
    ])

def _ec_chip(label, val_str, color=HEAD):
    return html.Div([
        html.Div(label, style={"fontSize": "8px", "color": MUTED, "fontWeight": "700",
                                "textTransform": "uppercase", "letterSpacing": "0.10em",
                                "marginBottom": "4px"}),
        html.Div(val_str, style={"fontSize": "18px", "fontWeight": "700", "color": color,
                                  "fontFamily": FONT_BODY}),
    ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "10px 12px",
              "border": BORDER_W})

@app.callback(
    Output("ec-kpi-mean","children"), Output("ec-kpi-rms","children"),
    Output("ec-kpi-variance","children"), Output("ec-kpi-max","children"),
    Output("ec-kpi-ber","children"), Output("ec-kpi-state","children"),
    Input("interval","n_intervals"),
)
def update_ec_kpis(_):
    with ec_lock: d = dict(ec_latest)
    if not d:
        return (_ec_chip(l,"--") for l in ["MEAN","RMS","VAR","MAX","BER","STATE"])
    feats = d.get("features", {})
    sv    = int(d.get("state", 0))
    sc    = DANGER if sv else OK
    return (
        _ec_chip("MEAN",     f"{feats.get('mean',0):.5f}"),
        _ec_chip("RMS",      f"{feats.get('rms',0):.5f}"),
        _ec_chip("VARIANCE", f"{feats.get('variance',0):.6f}"),
        _ec_chip("MAX",      f"{feats.get('max',0):.5f}"),
        _ec_chip("BER",      f"{d.get('ber',0):.4f}", WARN),
        _ec_chip("STATE",    "ANOMALY" if sv else "OK", sc),
    )

@app.callback(Output("ec-ber-chart","figure"), Input("interval","n_intervals"))
def update_ber_chart(_):
    with ec_lock: ts, bers = list(ec_history["ts"]), list(ec_history["ber"])
    fig = go.Figure()
    if ts:
        fig.add_trace(go.Scatter(x=ts, y=bers, mode="lines",
                                  line={"color": ACC2, "width": 1.5},
                                  fill="tozeroy", fillcolor="rgba(204,204,204,0.07)",
                                  name="BER"))
    fig.update_layout(**_chart_layout(
        yaxis={"gridcolor": GRID, "zeroline": True, "zerolinecolor": GRID2, "title": "BER"}))
    return fig

@app.callback(Output("ec-state-chart","figure"), Input("interval","n_intervals"))
def update_state_chart(_):
    with ec_lock: ts, states = list(ec_history["ts"]), list(ec_history["state"])
    fig = go.Figure()
    if ts:
        colors = [DANGER if s else OK for s in states]
        fig.add_trace(go.Scatter(x=ts, y=states, mode="markers+lines",
                                  marker={"color": colors, "size": 5},
                                  line={"color": GRID2, "width": 0.8}, name="State"))
    fig.update_layout(**_chart_layout(
        yaxis={"gridcolor": GRID, "tickvals": [0, 1], "ticktext": ["OK", "ANOMALY"],
               "range": [-0.3, 1.4], "zeroline": False}))
    return fig

@app.callback(Output("ec-raw-record","children"), Input("interval","n_intervals"))
def update_ec_raw(_):
    with ec_lock: d = dict(ec_latest)
    return json.dumps(d, indent=2) if d else "Waiting for EC data..."

# ─── TAB: EVENT LOG ───────────────────────────────────────────────────────────
def build_event_log():
    th_s = {"fontSize": "8px", "fontWeight": "700", "color": MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.08em",
            "padding": "6px 10px", "borderBottom": f"1px solid {GRID}",
            "textAlign": "left", "fontFamily": FONT_BODY}

    fail_rows = []
    for e in reversed(list(failure_log)):
        fail_rows.append(html.Tr([
            html.Td(e["ts"], style={"color": MUTED, "fontFamily": FONT_BODY, "fontSize": "10px",
                                     "padding": "4px 10px", "whiteSpace": "nowrap"}),
            html.Td(MACHINE_NAMES.get(e["station"], e["station"]),
                    style={"color": HEAD, "fontWeight": "700", "fontSize": "10px",
                           "padding": "4px 10px"}),
            html.Td(e["station"], style={"color": MUTED2, "fontFamily": FONT_BODY,
                                          "fontSize": "9px", "padding": "4px 10px"}),
            html.Td(e["label"], style={"color": DANGER, "fontFamily": FONT_BODY,
                                        "fontSize": "10px", "padding": "4px 10px"}),
        ], style={"borderBottom": f"1px solid {GRID}"}))
    if not fail_rows:
        fail_rows = [html.Tr([html.Td("No anomalies recorded yet.", colSpan=4,
                                       style={"color": MUTED, "padding": "14px",
                                              "textAlign": "center", "fontSize": "11px"})])]

    gantt_rows = []
    for t in reversed(list(gantt_tasks)[-50:]):
        dur = (t["end"] - t["start"]).total_seconds() if t["start"] else 0
        gantt_rows.append(html.Tr([
            html.Td(str(t["start"])[-12:] if t["start"] else "--",
                    style={"color": MUTED, "fontFamily": FONT_BODY, "fontSize": "10px",
                           "padding": "3px 10px", "whiteSpace": "nowrap"}),
            html.Td(MACHINE_NAMES.get(t["station"], t["station"]),
                    style={"color": HEAD, "fontWeight": "700", "fontSize": "10px",
                           "padding": "3px 10px"}),
            html.Td(t["task"],
                    style={"color": TEXT, "fontFamily": FONT_BODY, "fontSize": "10px",
                           "padding": "3px 10px"}),
            html.Td(t["sub_task"],
                    style={"color": MUTED2, "fontFamily": FONT_BODY, "fontSize": "9px",
                           "padding": "3px 10px"}),
            html.Td(f"{dur:.1f}s",
                    style={"color": WHITE, "fontFamily": FONT_BODY, "fontSize": "10px",
                           "padding": "3px 10px", "textAlign": "right"}),
        ], style={"borderBottom": f"1px solid {GRID}"}))
    if not gantt_rows:
        gantt_rows = [html.Tr([html.Td("No completed tasks yet.", colSpan=5,
                                        style={"color": MUTED, "padding": "14px",
                                               "textAlign": "center", "fontSize": "11px"})])]

    return html.Div([
        section_hdr("Event Log — Anomalies & Task History", MUTED2),
        html.Div([
            section_hdr("Anomaly / Fault Log", DANGER),
            html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Timestamp", style=th_s),
                        html.Th("Machine",   style=th_s),
                        html.Th("ID",        style=th_s),
                        html.Th("Fault",     style=th_s),
                    ])),
                    html.Tbody(fail_rows),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
            ], style={"maxHeight": "200px", "overflowY": "auto"}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "marginBottom": "12px", "border": BORDER_W}),

        html.Div([
            section_hdr("Completed Sub-task Timeline", WHITE),
            html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Time",     style=th_s),
                        html.Th("Machine",  style=th_s),
                        html.Th("Task",     style=th_s),
                        html.Th("Sub-task", style=th_s),
                        html.Th("Duration", style={**th_s, "textAlign": "right"}),
                    ])),
                    html.Tbody(gantt_rows),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
            ], style={"maxHeight": "340px", "overflowY": "auto"}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "border": BORDER_W}),
    ])

# ─── TAB: MACHINE HEALTH ──────────────────────────────────────────────────────
def build_health():
    return html.Div([
        section_hdr("Machine Health & Deterioration Monitor", MUTED2),
        html.Div(id="health-score-cards",
                 style={"display": "flex", "gap": "8px", "flexWrap": "wrap",
                        "marginBottom": "14px"}),
        html.Div([
            html.Div([
                dcc.Graph(id="health-fleet-gauge", config={"displayModeBar": False},
                          style={"height": "230px"}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "1", "minWidth": "230px", "border": BORDER_W}),
            html.Div([
                section_hdr("Health Comparison — All Machines", MUTED2),
                dcc.Graph(id="health-bar-chart", config={"displayModeBar": False},
                          style={"height": "210px"}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "2", "minWidth": "300px", "border": BORDER_W}),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}),
        html.Div([
            section_hdr("Health Deterioration Trend — All Machines", MUTED2),
            dcc.Graph(id="health-trend-chart", config={"displayModeBar": False},
                      style={"height": "260px"}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "marginBottom": "12px", "border": BORDER_W}),
        html.Div([
            section_hdr("Score Breakdown — Sub-component Detail", MUTED2),
            html.Div(id="health-detail-table", style={"overflowX": "auto"}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "border": BORDER_W}),
    ])

@app.callback(Output("health-score-cards","children"), Input("interval","n_intervals"))
def update_health_cards(_):
    with health_lock:
        scores = dict(health_scores)
        comps  = {s: dict(health_components[s]) for s in STATIONS}
    cards = []
    for s in STATIONS:
        sc   = scores[s]
        col  = _health_color(sc)
        lbl  = _health_label(sc)
        c    = comps[s]
        sub_bars = []
        for label, key in [("Fault","fault"),("Motor","motor"),("Duration","duration")]:
            val = c[key]
            vc  = _health_color(val)
            sub_bars.append(html.Div([
                html.Span(label, style={"fontSize": "8px", "color": MUTED,
                                         "fontWeight": "700", "textTransform": "uppercase",
                                         "letterSpacing": "0.06em", "minWidth": "52px",
                                         "display": "inline-block"}),
                html.Div([
                    html.Div(style={"height": "3px", "background": vc,
                                    "width": f"{val:.0f}%", "transition": "width 0.5s"}),
                ], style={"background": GRID, "borderRadius": "1px", "height": "3px",
                          "flex": "1", "margin": "0 7px"}),
                html.Span(f"{val:.0f}%", style={"fontSize": "9px", "color": vc,
                                                  "minWidth": "30px", "textAlign": "right"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"}))
        cards.append(html.Div([
            html.Div([
                html.Div(MACHINE_NAMES[s], style={"fontSize": "11px", "fontWeight": "700",
                                                    "color": HEAD}),
                html.Div(s, style={"fontSize": "8px", "color": MUTED, "marginTop": "1px"}),
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Span(f"{sc:.0f}", style={"fontSize": "38px", "fontWeight": "700",
                                               "color": col, "lineHeight": "1"}),
                html.Span("%", style={"fontSize": "16px", "color": col, "marginLeft": "2px"}),
            ], style={"marginBottom": "5px"}),
            tag(lbl, color=col),
            html.Div([
                html.Div(style={"height": "5px", "background": col,
                                "width": f"{sc:.0f}%", "transition": "width 0.6s"}),
            ], style={"background": GRID, "borderRadius": "1px", "height": "5px",
                      "margin": "10px 0 12px"}),
            *sub_bars,
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "14px",
                  "flex": "1", "minWidth": "145px",
                  "border": BORDER_W, "borderTop": f"2px solid {col}"}))
    return cards

@app.callback(Output("health-fleet-gauge","figure"), Input("interval","n_intervals"))
def update_fleet_gauge(_):
    with health_lock: scores = dict(health_scores)
    avg = sum(scores.values()) / max(len(scores), 1)
    col = _health_color(avg)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg,
        delta={"reference": 100, "valueformat": ".1f",
               "decreasing": {"color": DANGER}, "increasing": {"color": OK}},
        number={"suffix": "%", "font": {"size": 30, "color": HEAD, "family": FONT_BODY},
                "valueformat": ".1f"},
        title={"text": "Fleet Health Index", "font": {"size": 11, "color": MUTED2,
                                                       "family": FONT_BODY}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": GRID,
                     "tickfont": {"size": 8, "color": MUTED}, "nticks": 6},
            "bar": {"color": col, "thickness": 0.24},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": DANGER + "12"},
                {"range": [30, 55], "color": WARN   + "10"},
                {"range": [55, 80], "color": WARN   + "08"},
                {"range": [80,100], "color": OK     + "0a"},
            ],
            "threshold": {"line": {"color": col, "width": 2}, "thickness": 0.8, "value": avg},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin={"l": 24, "r": 24, "t": 24, "b": 8}, height=230,
                      font={"color": MUTED, "family": FONT_BODY})
    return fig

@app.callback(Output("health-bar-chart","figure"), Input("interval","n_intervals"))
def update_health_bar(_):
    with health_lock: scores = dict(health_scores)
    labels = [MACHINE_NAMES[s] for s in STATIONS]
    vals   = [scores[s] for s in STATIONS]
    colors = [_health_color(v) for v in vals]
    fig    = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=vals, marker_color=colors, marker_line_width=0,
        text=[f"{v:.0f}%" for v in vals], textposition="outside",
        textfont={"size": 10, "color": HEAD, "family": FONT_BODY},
    ))
    for thresh, col, lbl in [(80, OK, "NOMINAL"), (CRITICAL_HEALTH_THRESHOLD, DANGER, "CRITICAL"),
                              (WARNING_HEALTH_THRESHOLD, WARN, "WARNING")]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=col, line_width=1,
                      annotation_text=lbl,
                      annotation_position="right",
                      annotation_font={"size": 8, "color": col, "family": FONT_BODY})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG,
        font={"color": MUTED, "size": 10, "family": FONT_BODY},
        margin={"l": 36, "r": 80, "t": 16, "b": 54}, height=210,
        xaxis={"gridcolor": GRID, "tickangle": -20, "tickfont": {"size": 9}},
        yaxis={"gridcolor": GRID, "range": [0, 115], "zeroline": False, "title": "Health %"},
        bargap=0.3, showlegend=False,
    )
    return fig

@app.callback(Output("health-trend-chart","figure"), Input("interval","n_intervals"))
def update_health_trend(_):
    fig = go.Figure()
    shades = [WHITE, "#cccccc", "#aaaaaa", "#888888", "#666666", "#444444", "#333333"]
    with health_lock:
        for i, s in enumerate(STATIONS):
            trend = list(health_trend[s])
            if len(trend) < 2: continue
            ts_list = [t[0] for t in trend]
            sc_list = [t[1] for t in trend]
            fig.add_trace(go.Scatter(
                x=ts_list, y=sc_list, mode="lines",
                name=MACHINE_NAMES[s],
                line={"color": shades[i % len(shades)], "width": 1.5},
            ))
    if not fig.data:
        fig.add_annotation(text="Press PLAY to start playback",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"color": MUTED, "size": 12,
                                                  "family": FONT_BODY})
    for thresh, col, lbl in [(80, OK, "NOMINAL >"),
                              (CRITICAL_HEALTH_THRESHOLD, DANGER, "CRITICAL >"),
                              (WARNING_HEALTH_THRESHOLD, WARN, "WARNING >")]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=col + "88", line_width=1,
                      annotation_text=lbl, annotation_position="right",
                      annotation_font={"size": 8, "color": col, "family": FONT_BODY})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG,
        font={"color": MUTED, "size": 10, "family": FONT_BODY},
        margin={"l": 40, "r": 80, "t": 14, "b": 32}, height=260,
        xaxis={"gridcolor": GRID, "zeroline": False},
        yaxis={"gridcolor": GRID, "zeroline": False, "range": [-5, 108],
               "title": "Health %"},
        legend={"orientation": "h", "y": 1.06,
                "font": {"size": 9, "color": MUTED2, "family": FONT_BODY},
                "bgcolor": "rgba(0,0,0,0)"},
        hovermode="x unified",
    )
    return fig

@app.callback(Output("health-detail-table","children"), Input("interval","n_intervals"))
def update_health_table(_):
    with health_lock:
        scores = dict(health_scores)
        comps  = {s: dict(health_components[s]) for s in STATIONS}
    th_s = {"fontSize": "8px", "fontWeight": "700", "color": MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.07em",
            "padding": "6px 10px", "borderBottom": f"1px solid {GRID}",
            "textAlign": "left", "fontFamily": FONT_BODY}
    rows = []
    for s in STATIONS:
        sc  = scores[s]
        c   = comps[s]
        col = _health_color(sc)
        fc  = sum(1 for e in failure_log if e["station"] == s)
        rows.append(html.Tr([
            html.Td([
                html.Div(MACHINE_NAMES[s], style={"fontWeight": "700", "color": HEAD,
                                                    "fontSize": "11px"}),
                html.Div(s, style={"fontSize": "8px", "color": MUTED}),
            ], style={"padding": "7px 10px"}),
            html.Td(f"{sc:.1f}%",
                    style={"padding": "7px 10px", "fontWeight": "700",
                           "color": col, "fontFamily": FONT_BODY, "fontSize": "12px"}),
            html.Td(_health_label(sc),
                    style={"padding": "7px 10px", "fontSize": "9px",
                           "fontWeight": "700", "color": col}),
            html.Td(f"{c['fault']:.1f}%",
                    style={"padding": "7px 10px", "color": _health_color(c["fault"]),
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(f"{c['motor']:.1f}%",
                    style={"padding": "7px 10px", "color": _health_color(c["motor"]),
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(f"{c['duration']:.1f}%",
                    style={"padding": "7px 10px", "color": _health_color(c["duration"]),
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(str(fc), style={"padding": "7px 10px",
                                     "color": DANGER if fc else MUTED,
                                     "fontFamily": FONT_BODY, "fontSize": "10px",
                                     "fontWeight": "700" if fc else "400"}),
        ], style={"borderBottom": f"1px solid {GRID}"}))
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Machine",          style=th_s),
            html.Th("Health",           style=th_s),
            html.Th("Status",           style=th_s),
            html.Th("Fault (45%)",      style=th_s),
            html.Th("Motor (30%)",      style=th_s),
            html.Th("Duration (25%)",   style=th_s),
            html.Th("Total Faults",     style=th_s),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})

# ─── TAB: PREDICTIVE ──────────────────────────────────────────────────────────
def build_predictive():
    return html.Div([
        section_hdr("Predictive Maintenance — Trend-Based Failure Risk Analysis", MUTED2),

        # Risk matrix cards
        html.Div(id="predictive-risk-cards",
                 style={"display": "flex", "gap": "8px", "flexWrap": "wrap",
                        "marginBottom": "14px"}),

        html.Div([
            # Failure risk bar
            html.Div([
                section_hdr("Failure Risk — All Machines", MUTED2),
                dcc.Graph(id="pred-risk-bar", config={"displayModeBar": False},
                          style={"height": "220px"}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "2", "minWidth": "300px", "border": BORDER_W}),

            # Anomaly score bars
            html.Div([
                section_hdr("Anomaly Score (Z-score) — Real-Time", MUTED2),
                dcc.Graph(id="pred-anomaly-bar", config={"displayModeBar": False},
                          style={"height": "220px"}),
            ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                      "flex": "1", "minWidth": "240px", "border": BORDER_W}),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}),

        # KPI aggregation table
        html.Div([
            section_hdr("KPI Aggregation — Per Machine", MUTED2),
            html.Div(id="predictive-kpi-table", style={"overflowX": "auto"}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "marginBottom": "12px", "border": BORDER_W}),

        # Alert history
        html.Div([
            section_hdr("Alert History — System Log", MUTED2),
            html.Div(id="predictive-alert-log",
                     style={"maxHeight": "250px", "overflowY": "auto"}),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "border": BORDER_W}),
    ])

@app.callback(
    Output("predictive-risk-cards","children"),
    Output("pred-risk-bar","figure"),
    Output("pred-anomaly-bar","figure"),
    Output("predictive-kpi-table","children"),
    Output("predictive-alert-log","children"),
    Input("interval","n_intervals"),
    Input("main-tabs","value"),
)
def update_predictive(_, tab):
    if tab != "predictive":
        raise PreventUpdate

    with health_lock:  scores = dict(health_scores)
    with prediction_lock:
        risks  = dict(failure_risk)
        ruls   = dict(predicted_rul)
        a_scores = dict(anomaly_score)

    # ── Risk cards ──
    cards = []
    for s in STATIONS:
        rk  = risks[s]
        rl  = ruls[s]
        sc  = scores[s]
        col = DANGER if rk > 75 else WARN if rk > 45 else OK
        hcol = _health_color(sc)
        kpi  = kpi_store[s]
        cards.append(html.Div([
            html.Div([
                html.Span(STATION_ICONS[s] + " ", style={"fontSize": "13px"}),
                html.Span(MACHINE_NAMES[s], style={"fontSize": "10px", "fontWeight": "700",
                                                    "color": HEAD}),
            ], style={"marginBottom": "8px"}),
            # Risk meter
            html.Div("FAILURE RISK", style={"fontSize": "7px", "color": MUTED,
                                             "letterSpacing": "0.10em", "marginBottom": "3px"}),
            html.Div([
                html.Div(style={"height": "5px", "background": col,
                                "width": f"{rk:.0f}%", "transition": "width 0.5s"}),
            ], style={"background": GRID, "height": "5px", "borderRadius": "1px",
                      "marginBottom": "4px"}),
            html.Div([
                html.Span(f"{rk:.0f}%", style={"fontSize": "24px", "fontWeight": "700",
                                                "color": col, "lineHeight": "1",
                                                "marginRight": "8px"}),
                html.Div([
                    html.Div(f"RUL: {rl:.0f}%" if rl is not None else "RUL: --",
                             style={"fontSize": "9px", "color": MUTED2}),
                    html.Div(f"Health: {sc:.0f}%",
                             style={"fontSize": "9px", "color": hcol}),
                ]),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div([
                html.Div(f"Faults: {kpi['faults_total']}",
                         style={"fontSize": "8px", "color": DANGER if kpi['faults_total'] else MUTED}),
                html.Div(f"Motor anomalies: {kpi['motor_anomalies']}",
                         style={"fontSize": "8px", "color": WARN if kpi['motor_anomalies'] else MUTED}),
                html.Div(f"Dur. exceedances: {kpi['duration_exceedances']}",
                         style={"fontSize": "8px", "color": MUTED2}),
            ]),
        ], style={"background": CARD2, "borderRadius": BORDER_R, "padding": "12px",
                  "flex": "1", "minWidth": "145px",
                  "border": BORDER_W, "borderTop": f"2px solid {col}"}))

    # ── Risk bar chart ──
    labels = [MACHINE_NAMES[s] for s in STATIONS]
    risk_vals = [risks[s] for s in STATIONS]
    r_colors  = [DANGER if v > 75 else WARN if v > 45 else OK for v in risk_vals]
    fig_risk  = go.Figure()
    fig_risk.add_trace(go.Bar(
        x=labels, y=risk_vals, marker_color=r_colors, marker_line_width=0,
        text=[f"{v:.0f}%" for v in risk_vals], textposition="outside",
        textfont={"size": 9, "color": HEAD, "family": FONT_BODY},
    ))
    fig_risk.add_hline(y=75, line_dash="dot", line_color=DANGER, line_width=1,
                       annotation_text="HIGH RISK",
                       annotation_font={"size": 8, "color": DANGER, "family": FONT_BODY})
    fig_risk.add_hline(y=45, line_dash="dot", line_color=WARN, line_width=1,
                       annotation_text="ELEVATED",
                       annotation_font={"size": 8, "color": WARN, "family": FONT_BODY})
    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG,
        font={"color": MUTED, "size": 9, "family": FONT_BODY},
        margin={"l": 36, "r": 80, "t": 14, "b": 50}, height=220,
        xaxis={"gridcolor": GRID, "tickangle": -15, "tickfont": {"size": 9}},
        yaxis={"gridcolor": GRID, "range": [0, 115], "zeroline": False, "title": "Risk %"},
        bargap=0.3, showlegend=False,
    )

    # ── Anomaly score bar ──
    a_vals   = [a_scores[s] for s in STATIONS]
    a_colors = [DANGER if v > 75 else WARN if v > 40 else MUTED2 for v in a_vals]
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Bar(
        x=[MACHINE_NAMES[s] for s in STATIONS], y=a_vals,
        marker_color=a_colors, marker_line_width=0,
        text=[f"{v:.0f}" for v in a_vals], textposition="outside",
        textfont={"size": 9, "color": HEAD, "family": FONT_BODY},
    ))
    fig_anom.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG,
        font={"color": MUTED, "size": 9, "family": FONT_BODY},
        margin={"l": 36, "r": 20, "t": 14, "b": 50}, height=220,
        xaxis={"gridcolor": GRID, "tickangle": -15, "tickfont": {"size": 9}},
        yaxis={"gridcolor": GRID, "range": [0, 115], "zeroline": False,
               "title": "Anomaly Score"},
        bargap=0.3, showlegend=False,
    )

    # ── KPI table ──
    th_s = {"fontSize": "8px", "fontWeight": "700", "color": MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.07em",
            "padding": "6px 10px", "borderBottom": f"1px solid {GRID}",
            "textAlign": "left", "fontFamily": FONT_BODY}
    kpi_rows = []
    for s in STATIONS:
        kpi  = kpi_store[s]
        sc   = scores[s]
        hcol = _health_color(sc)
        kpi_rows.append(html.Tr([
            html.Td([
                html.Div(MACHINE_NAMES[s], style={"fontWeight": "700", "color": HEAD,
                                                    "fontSize": "10px"}),
            ], style={"padding": "6px 10px"}),
            html.Td(f"{sc:.0f}%",
                    style={"padding": "6px 10px", "color": hcol,
                           "fontFamily": FONT_BODY, "fontSize": "11px", "fontWeight": "700"}),
            html.Td(f"{risks[s]:.0f}%",
                    style={"padding": "6px 10px", "fontFamily": FONT_BODY, "fontSize": "10px",
                           "color": DANGER if risks[s]>75 else WARN if risks[s]>45 else OK}),
            html.Td(str(kpi["faults_total"]),
                    style={"padding": "6px 10px", "color": DANGER if kpi["faults_total"] else MUTED,
                           "fontFamily": FONT_BODY, "fontSize": "10px",
                           "fontWeight": "700" if kpi["faults_total"] else "400"}),
            html.Td(str(kpi["motor_anomalies"]),
                    style={"padding": "6px 10px", "color": WARN if kpi["motor_anomalies"] else MUTED,
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(str(kpi["stalls"]),
                    style={"padding": "6px 10px", "color": WARN if kpi["stalls"] else MUTED,
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(str(kpi["duration_exceedances"]),
                    style={"padding": "6px 10px", "color": MUTED2,
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(f"{kpi['avg_task_dur']:.1f}s",
                    style={"padding": "6px 10px", "color": TEXT,
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(str(kpi["tasks_completed"]),
                    style={"padding": "6px 10px", "color": MUTED2,
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
            html.Td(f"{a_scores[s]:.0f}",
                    style={"padding": "6px 10px",
                           "color": DANGER if a_scores[s]>75 else WARN if a_scores[s]>40 else MUTED,
                           "fontFamily": FONT_BODY, "fontSize": "10px"}),
        ], style={"borderBottom": f"1px solid {GRID}"}))

    kpi_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Machine",        style=th_s),
            html.Th("Health",         style=th_s),
            html.Th("Fail Risk",      style=th_s),
            html.Th("Faults",         style=th_s),
            html.Th("Motor Anom.",    style=th_s),
            html.Th("Stalls",         style=th_s),
            html.Th("Dur. Exceed.",   style=th_s),
            html.Th("Avg Task Dur",   style=th_s),
            html.Th("Tasks Done",     style=th_s),
            html.Th("Anomaly Score",  style=th_s),
        ])),
        html.Tbody(kpi_rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    # ── Alert log ──
    with alerts_lock:
        all_alerts = list(active_alerts)
    alert_rows = []
    for a in reversed(all_alerts[-40:]):
        col = SEV_COLOR.get(a["severity"], MUTED2)
        alert_rows.append(html.Div([
            html.Span(f"[{a['severity'].upper()}]",
                      style={"color": col, "fontWeight": "700", "minWidth": "72px",
                             "display": "inline-block", "fontSize": "9px",
                             "letterSpacing": "0.07em"}),
            html.Span(str(a["ts"])[-15:],
                      style={"color": MUTED, "marginRight": "10px", "fontSize": "9px",
                             "minWidth": "90px", "display": "inline-block"}),
            html.Span(MACHINE_NAMES.get(a["station"], a["station"]),
                      style={"color": WHITE, "marginRight": "10px", "fontWeight": "600",
                             "fontSize": "10px", "minWidth": "140px",
                             "display": "inline-block"}),
            html.Span(a["msg"], style={"color": TEXT, "fontSize": "10px"}),
        ], style={"padding": "4px 0", "borderBottom": f"1px solid {GRID}",
                  "fontFamily": FONT_BODY, "display": "flex", "alignItems": "center"}))
    if not alert_rows:
        alert_rows = [html.Div("No alerts generated yet.",
                                style={"color": MUTED, "fontSize": "11px",
                                       "padding": "10px 0", "fontFamily": FONT_BODY})]

    return cards, fig_risk, fig_anom, kpi_table, alert_rows

# ─── TAB: DRILL-DOWN ──────────────────────────────────────────────────────────
def build_drill_down():
    return html.Div([
        section_hdr("Drill-Down — Select a Machine", MUTED2),
        html.Div([
            html.Span("STATION:", style={"fontSize": "9px", "color": MUTED,
                                          "marginRight": "10px", "letterSpacing": "0.10em"}),
            dcc.Dropdown(id="drill-station-select",
                         options=[{"label": MACHINE_NAMES[s], "value": s} for s in STATIONS],
                         value=None, clearable=False,
                         style={"width": "260px", "background": CARD2, "color": HEAD,
                                "border": f"1px solid {GRID2}", "borderRadius": BORDER_R,
                                "fontFamily": FONT_BODY, "fontSize": "11px"}),
        ], style={"marginBottom": "12px", "display": "flex", "alignItems": "center"}),
        html.Div(id="drill-down-content",
                 style={"background": CARD, "borderRadius": BORDER_R,
                        "minHeight": "280px"}),
    ])

@app.callback(
    Output("drill-down-content","children"),
    Input("drill-station-select","value"),
    Input("interval","n_intervals"),
)
def show_drill_down(station, _):
    if not station:
        return html.Div("Select a machine to view full details.",
                        style={"textAlign": "center", "fontSize": "13px",
                               "color": MUTED, "paddingTop": "60px",
                               "fontFamily": FONT_BODY})
    return render_station_detail(station)

# ─── PLAYBACK ENGINE ──────────────────────────────────────────────────────────
playback = {
    "running":      False,
    "speed":        5.0,
    "current_line": 0,
    "total_lines":  0,
    "log_file":     None,
    "paused":       False,
}
playback_lock = threading.Lock()

def run_playback(filepath: str):
    import re as _re
    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    playback["total_lines"] = len(lines)
    prev_ts = None

    for idx, line in enumerate(lines):
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
        ts_str  = obj.get("timestamp", "")
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                ts = datetime.now()

        if prev_ts is not None:
            delta = (ts - prev_ts).total_seconds()
            if delta > 0:
                time.sleep(max(0, delta / playback["speed"]))
        prev_ts = ts

        update_ec_state(obj, ts)

        if station not in STATIONS:
            continue

        with playback_lock:
            latest[station] = obj
            playback["current_line"] = idx + 1

            for key in ("m1_speed", "m2_speed", "m3_speed", "m4_speed"):
                if key in obj:
                    motor_history[station][key].append((ts, obj[key]))

            if station == "VGR_1":
                pos_history["VGR_1"]["x"].append(obj.get("current_pos_x", 0))
                pos_history["VGR_1"]["y"].append(obj.get("current_pos_y", 0))
                pos_history["VGR_1"]["z"].append(obj.get("current_pos_z", 0))
                pos_history["VGR_1"]["ts"].append(ts)
            if station == "HBW_1":
                pos_history["HBW_1"]["x"].append(obj.get("current_pos_x", 0))
                pos_history["HBW_1"]["y"].append(obj.get("current_pos_y", 0))
                pos_history["HBW_1"]["ts"].append(ts)

            _compute_health(station, obj, ts)

            fl = obj.get("failure_label", "")
            if fl:
                failure_log.append({"ts": ts_str, "station": station, "label": fl})

            sub_or_task = (obj.get("current_sub_task") or obj.get("current_task") or "").lower()
            resolved    = "idle"
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
                "task":       obj.get("current_task", ""),
                "sub":        obj.get("current_sub_task", ""),
                "elapsed":    obj.get("current_task_duration", 0.0),
                "state":      obj.get("current_state", ""),
            })
            if resolved != prev_label and resolved != "idle":
                hist = sm_state[station]["history"]
                if not hist or hist[-1] != resolved:
                    hist.append(resolved)
                    if len(hist) > 12:
                        hist.pop(0)

            cur_task = obj.get("current_task") or ""
            cur_sub  = obj.get("current_sub_task") or ""
            active   = _active_sub[station]
            label    = cur_sub if cur_sub else cur_task
            if label != active["sub"] or cur_task != active["task"]:
                if active["sub"] and active["start"] is not None:
                    gantt_tasks.append({
                        "station":  station,
                        "task":     active["task"],
                        "sub_task": active["sub"],
                        "start":    active["start"],
                        "end":      ts,
                    })
                _active_sub[station] = {"task": cur_task, "sub": label,
                                        "start": ts if label else None}

    playback["running"] = False

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Factory Predictive Dashboard")
    parser.add_argument("--file", "-f", default="low-level_log_20230206-140808.txt")
    parser.add_argument("--port", "-p", type=int, default=8050)
    args = parser.parse_args()

    log_path = Path(args.file)
    if not log_path.exists():
        print(f"ERROR: File not found: {log_path}")
        exit(1)

    playback["log_file"] = str(log_path)
    print(f"""
╔══════════════════════════════════════════════════════╗
║  SMART FACTORY · PREDICTIVE MONITOR                  ║
║  Log  : {log_path.name:<44}║
║  URL  : http://127.0.0.1:{args.port:<27}║
║  Press [PLAY] in the browser to begin                ║
╚══════════════════════════════════════════════════════╝
    """)
    app.run(debug=False, port=args.port)
