"""
pipeline.py — TELE 523 · Group 1
Master integration pipeline: runs every stage end-to-end.

Stages
------
1. Preprocessing          (src/preprocessing.py)
2. Signal Processing      (src/signal_processing.py)
3. Modulation             (src/modulation.py  → modulation.main())
4. Demodulation           (src/demodulation.py → demodulation.main())
5. Digital Telemetry      (src/digital_telemetry.py → generate_log())
6. Dashboard              (src/dashboard.py  → Dash web app on port 8050)

Usage
-----
    python pipeline.py              # run all stages
    python pipeline.py --stage 1 3  # run only stages 1 and 3 (1-indexed)
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
SRC_DIR  = BASE_DIR / "src"

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _banner(text):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def _ok(msg):   print(f"  {GREEN}✓  {msg}{RESET}")
def _warn(msg): print(f"  {YELLOW}⚠  {msg}{RESET}")
def _err(msg):  print(f"  {RED}✗  {msg}{RESET}")
def _info(msg): print(f"  {CYAN}→  {msg}{RESET}")


# ── Stage runner helpers ───────────────────────────────────────────────────────

def _run_script(script_path: Path, stage_name: str) -> bool:
    """Run a Python script as a subprocess and report success/failure."""
    _info(f"Running {script_path.name} ...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True,
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        _ok(f"{stage_name} complete  ({elapsed:.1f}s)")
        return True
    else:
        _err(f"{stage_name} FAILED  (exit code {result.returncode})")
        return False


def _run_module_main(module_name: str, stage_name: str) -> bool:
    """Import a src/ module and call its main() or generate_log() function."""
    # Add src to path temporarily
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)

    _info(f"Importing {module_name} ...")
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_name)
        if hasattr(mod, "main"):
            mod.main()
        elif hasattr(mod, "generate_log"):
            mod.generate_log()
        else:
            _warn(f"{module_name} has no main() or generate_log() — skipped")
            return False
        elapsed = time.time() - t0
        _ok(f"{stage_name} complete  ({elapsed:.1f}s)")
        return True
    except Exception as exc:
        _err(f"{stage_name} FAILED: {exc}")
        return False


# ── Individual stages ──────────────────────────────────────────────────────────

def stage_preprocessing() -> bool:
    _banner("Stage 1 · Preprocessing")
    return _run_script(SRC_DIR / "preprocessing.py", "Preprocessing")


def stage_signal_processing() -> bool:
    _banner("Stage 2 · Signal Processing")
    return _run_script(SRC_DIR / "signal_processing.py", "Signal Processing")


def stage_modulation() -> bool:
    _banner("Stage 3 · Modulation")
    return _run_module_main("modulation", "Modulation")


def stage_demodulation() -> bool:
    _banner("Stage 4 · Demodulation")
    return _run_module_main("demodulation", "Demodulation")


def stage_digital_telemetry() -> bool:
    _banner("Stage 5 · Digital Telemetry")
    return _run_module_main("digital_telemetry", "Digital Telemetry")


def stage_dashboard() -> bool:
    _banner("Stage 6 · Dashboard")
    log_file = BASE_DIR / "low-level_log_20230206-140808.txt"
    if not log_file.exists():
        _err(f"Log file not found: {log_file}")
        return False
    _info("Launching Dash dashboard on http://127.0.0.1:8050 (press Ctrl+C to stop)")
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "dashboard.py"), "--file", str(log_file), "--port", "8050"],
        capture_output=False,
        text=True,
        cwd=str(BASE_DIR),
    )
    if result.returncode == 0:
        _ok("Dashboard exited cleanly.")
        return True
    else:
        _err(f"Dashboard exited with code {result.returncode}")
        return False


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    _banner("Pipeline Summary")
    all_ok = True
    for stage, ok in results.items():
        if ok:
            _ok(stage)
        else:
            _err(stage)
            all_ok = False
    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}All stages passed.{RESET}\n")
    else:
        print(f"  {YELLOW}{BOLD}Some stages failed — check output above.{RESET}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

STAGE_MAP = {
    1: ("Preprocessing",      stage_preprocessing),
    2: ("Signal Processing",  stage_signal_processing),
    3: ("Modulation",         stage_modulation),
    4: ("Demodulation",       stage_demodulation),
    5: ("Digital Telemetry",  stage_digital_telemetry),
    6: ("Dashboard",          stage_dashboard),
}


def main():
    parser = argparse.ArgumentParser(
        description="TELE 523 Group 1 — End-to-end telemetry pipeline"
    )
    parser.add_argument(
        "--stage", nargs="+", type=int, metavar="N",
        help="Run only specific stages by number (1=Preprocessing … 6=Dashboard)."
    )
    args = parser.parse_args()

    print(f"\n{BOLD}TELE 523 · Group 1 · Industrial Machine Condition Monitoring{RESET}")
    print(f"{BOLD}End-to-End Telemetry Pipeline{RESET}\n")

    results = {}

    stages_to_run = args.stage if args.stage else list(STAGE_MAP.keys())
    invalid = [s for s in stages_to_run if s not in STAGE_MAP]
    if invalid:
        _err(f"Unknown stage numbers: {invalid}. Valid: 1–6")
        sys.exit(1)

    for num in stages_to_run:
        name, fn = STAGE_MAP[num]
        results[f"Stage {num}: {name}"] = fn()

    print_summary(results)


if __name__ == "__main__":
    main()
