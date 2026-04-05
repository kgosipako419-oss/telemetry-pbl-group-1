"""
main.py — Student 3: Modulation Lead
TELE 523 · Group 1 · Industrial Machine Condition Monitoring

Single entry point. Runs the full modulation phase in 4 steps:

  Step 1 — modulation.py                   : AM/FM/ASK/FSK/PSK + metrics
  Step 2 — generate_modulation_outputs.py  : 70 modulated signal CSVs
  Step 3 — demodulation.py                 : demodulation metrics
  Step 4 — generate_demodulation_outputs.py: 70 demodulated handoff CSVs

Inputs  (data/processed/): 7 × _filtered.csv + 7 × _features.csv  = 14 files
Outputs (results/...):     70 modulated + 70 demodulated            = 140 files

Usage:
  python src/main.py
"""

import os, sys, time, traceback

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

BOLD  = "\033[1m"; GREEN = "\033[92m"; CYAN  = "\033[96m"
YELLOW= "\033[93m"; RED  = "\033[91m"; RESET = "\033[0m"; DIM = "\033[2m"

def banner(t):
    print(f"\n{BOLD}{CYAN}{'='*65}\n  {t}\n{'='*65}{RESET}\n")

def step_hdr(n, t):
    print(f"\n{BOLD}{YELLOW}[ STEP {n} ] {t}{RESET}")
    print(f"{YELLOW}{'-'*55}{RESET}")

def ok(t):   print(f"{GREEN}  ✔  {t}{RESET}")
def fail(t): print(f"{RED}  ✘  {t}{RESET}")
def elapsed(s): return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {s%60:.0f}s"


def check_inputs():
    from modulation import STATIONS, PROCESSED_PATH
    missing = []
    for s in STATIONS:
        for suffix in ["_filtered.csv", "_features.csv"]:
            f = os.path.join(PROCESSED_PATH, f"{s}{suffix}")
            if not os.path.exists(f):
                missing.append(f"{s}{suffix}")
    if missing:
        fail("Missing input files from Student 2:")
        for m in missing: print(f"       data/processed/{m}")
        print("\n  Run preprocessing.py and signal_processing.py first.\n")
        sys.exit(1)
    ok(f"All 14 input files found (7 × _filtered + 7 × _features)")


def run_modulation():
    import modulation
    modulation.BASE_DIR       = BASE_DIR
    modulation.PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
    modulation.OUT_DIR        = os.path.join(BASE_DIR, "results", "modulation")
    os.makedirs(modulation.OUT_DIR, exist_ok=True)
    modulation.main()


def run_generate_modulation_outputs():
    import generate_modulation_outputs as gmo
    gmo.BASE_DIR = BASE_DIR
    gmo.OUT_DIR  = os.path.join(BASE_DIR, "results", "modulation", "output")
    os.makedirs(gmo.OUT_DIR, exist_ok=True)
    gmo.generate_outputs()


def run_demodulation():
    import demodulation
    demodulation.BASE_DIR = BASE_DIR
    demodulation.OUT_DIR  = os.path.join(BASE_DIR, "results", "demodulation")
    os.makedirs(demodulation.OUT_DIR, exist_ok=True)
    demodulation.main()


def run_generate_demodulation_outputs():
    import generate_demodulation_outputs as gdo
    gdo.BASE_DIR = BASE_DIR
    gdo.OUT_DIR  = os.path.join(BASE_DIR, "results", "demodulation", "output")
    os.makedirs(gdo.OUT_DIR, exist_ok=True)
    gdo.generate_demodulation_outputs()


def main():
    banner("TELE 523 · Student 3 — Modulation Phase\n  "
           "Inputs: 14 files (7 filtered + 7 features)\n  "
           "Outputs: 70 modulated + 70 demodulated = 140 files")

    total_start = time.time()
    print("Checking inputs...")
    check_inputs()

    steps = [
        (1, "Modulation — AM/FM/ASK/FSK/PSK on all 14 input files",
         run_modulation),
        (2, "Generate modulation output CSVs (70 files)",
         run_generate_modulation_outputs),
        (3, "Demodulation — recover all signals + metrics",
         run_demodulation),
        (4, "Generate demodulation output CSVs (70 files → Student 4)",
         run_generate_demodulation_outputs),
    ]

    log = []
    for n, desc, fn in steps:
        step_hdr(n, desc)
        t0 = time.time()
        try:
            fn()
            dur = elapsed(time.time() - t0)
            ok(f"Step {n} complete  ({dur})")
            log.append((n, desc, "OK", dur))
        except Exception as e:
            dur = elapsed(time.time() - t0)
            fail(f"Step {n} failed  ({dur})")
            traceback.print_exc()
            log.append((n, desc, "FAILED", dur))
            print(f"\n{RED}  Pipeline stopped at Step {n}. Fix the error above and rerun.{RESET}\n")
            sys.exit(1)

    banner(f"Pipeline Complete — total time: {elapsed(time.time()-total_start)}")

    print(f"  {'Step':<6} {'Status':<8} {'Time':<8} Description")
    print(f"  {'-'*62}")
    for n, desc, status, dur in log:
        col = GREEN if status=="OK" else RED
        print(f"  {n:<6} {col}{status:<8}{RESET} {dur:<8} {desc}")

    demod_out = os.path.join(BASE_DIR, "results", "demodulation", "output")
    print(f"""
  Output summary:
    70 modulated CSVs      ->  results/modulation/output/
    70 demodulated CSVs    ->  results/demodulation/output/
    Metrics                ->  results/modulation/modulation_results.csv
                               results/demodulation/demodulation_metrics.csv

{BOLD}  Student 4 handoff files (70):
    {{STATION}}_filtered_AM_demod.csv   / {{STATION}}_features_AM_demod.csv
    {{STATION}}_filtered_FM_demod.csv   / {{STATION}}_features_FM_demod.csv
    {{STATION}}_filtered_ASK_demod.csv  / {{STATION}}_features_ASK_demod.csv
    {{STATION}}_filtered_FSK_demod.csv  / {{STATION}}_features_FSK_demod.csv
    {{STATION}}_filtered_PSK_demod.csv  / {{STATION}}_features_PSK_demod.csv{RESET}
""")


if __name__ == "__main__":
    main()