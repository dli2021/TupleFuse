"""End-to-end reproduction driver for the TupleFuse experiments.

Run from the package root after installing requirements and downloading the
external data (see README):

    python code/run_all.py            # preprocessing + all CPU experiments
    python code/run_all.py --timing   # timing experiments only (run alone)

The two phases are separated on purpose: the timing experiments (rows, single
host partition/thread scaling, tuple width) measure wall-clock throughput and
should run on an otherwise idle machine, while the first phase reproduces every
accuracy, determinism, budget, risk, drift, and portability result.
"""
import os
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

PREP = ["prep2017.py", "prep_leipzig.py", "prep_magellan.py"]

MAIN = ["exp01_quality.py", "exp02_budget.py", "exp03_heterogeneity.py",
        "exp04_riskcoverage.py", "exp05_drift.py", "exp06_determinism.py",
        "exp10_sourcecount.py", "exp11_streaming.py", "exp12_portability.py"]

# Timing experiments. exp07 includes a Spark local-mode cell (needs a JDK);
# drop the "--spark" argument to skip it.
TIMING = [("exp07_scal_rows.py", ["--spark"]),
          ("exp08_scal_partitions.py", []),
          ("exp09_tuplewidth.py", [])]


def run(script, args=()):
    print(f"=== {script} {' '.join(args)} ===", flush=True)
    r = subprocess.run([PY, os.path.join(SRC, script), *args])
    if r.returncode != 0:
        print(f"!!! {script} failed with code {r.returncode}", flush=True)
    return r.returncode


def main():
    if "--timing" in sys.argv:
        for s, a in TIMING:
            run(s, a)
    else:
        for s in PREP + MAIN:
            run(s)


if __name__ == "__main__":
    main()
