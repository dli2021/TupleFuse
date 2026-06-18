"""Experiment 7: row scalability and engine portability. Each (engine, size)
runs in its own subprocess for clean peak-memory and isolated timing."""
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, save_csv  # noqa: E402
import synth  # noqa: E402

PY = sys.executable
SRC = os.path.dirname(os.path.abspath(__file__))
SIZES = [1, 2, 5, 10, 20, 50]          # millions
ENGINES = ["pandas", "polars", "duckdb"]
THREADS = 16


def run_worker(engine, n_rows, threads, width=4, timeout=1800):
    cmd = [PY, os.path.join(SRC, "scal_worker.py"), engine, str(n_rows),
           str(threads), str(width)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"engine": engine, "n_rows": n_rows, "error": "timeout"}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"engine": engine, "n_rows": n_rows, "error":
            (r.stderr or r.stdout)[-400:]}


def main():
    synth.load_scal_base()  # ensure base exists before timing
    spark_ok = "--spark" in sys.argv
    engines = ENGINES + (["spark"] if spark_ok else [])
    rows = []
    for m in SIZES:
        n = m * 1_000_000
        for eng in engines:
            if eng == "spark" and m > 10:
                continue  # JVM driver memory bound on 16 GB host
            rec = run_worker(eng, n, THREADS)
            rows.append(rec)
            print(rec, flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "exp07_scal_rows.csv")


if __name__ == "__main__":
    main()
