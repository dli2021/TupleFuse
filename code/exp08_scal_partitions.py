"""Experiment 8: single-host partition/thread scaling at fixed workload.
multiprocessing: critical-partition time + wall time; DuckDB/Polars: threads."""
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_csv  # noqa: E402
import synth  # noqa: E402

PY = sys.executable
SRC = os.path.dirname(os.path.abspath(__file__))
SIZES_M = [10, 20]
PARTS = [1, 2, 4, 8, 12, 16]


def run_worker(engine, n_rows, threads, timeout=2400):
    cmd = [PY, os.path.join(SRC, "scal_worker.py"), engine, str(n_rows),
           str(threads), "4"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"engine": engine, "n_rows": n_rows, "error": "timeout"}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"engine": engine, "n_rows": n_rows,
            "error": (r.stderr or r.stdout)[-400:]}


def main():
    synth.load_scal_base()
    rows = []
    for m in SIZES_M:
        n = m * 1_000_000
        for P in PARTS:
            rec = run_worker(f"mp{P}", n, min(P, 16))
            rec["mode"] = "multiprocessing"
            rec["partitions"] = P
            rows.append(rec)
            print(rec, flush=True)
        for P in PARTS:
            rec = run_worker("duckdb", n, P)
            rec["mode"] = "duckdb-threads"
            rec["partitions"] = P
            rows.append(rec)
            print(rec, flush=True)
        for P in PARTS:
            rec = run_worker("polars", n, P)
            rec["mode"] = "polars-threads"
            rec["partitions"] = P
            rows.append(rec)
            print(rec, flush=True)
    df = pd.DataFrame(rows)
    # speedups relative to P=1 within each (mode, n_rows)
    out = []
    for (mode, n), g in df.groupby(["mode", "n_rows"]):
        g = g.sort_values("partitions").copy()
        if mode == "multiprocessing":
            base = g[g.partitions == 1]["critical_s"].iloc[0]
            g["speedup"] = base / g["critical_s"]
            g["wall_speedup"] = g[g.partitions == 1]["wall_s"].iloc[0] / g["wall_s"]
        else:
            base = g[g.partitions == 1]["time_s"].iloc[0]
            g["speedup"] = base / g["time_s"]
        g["efficiency"] = g["speedup"] / g["partitions"]
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    save_csv(df, "exp08_scal_partitions.csv")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
