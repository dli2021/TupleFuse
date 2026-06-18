"""Subprocess worker: run ONE (engine, n_rows, threads) scalability measurement
and print a single JSON line. Isolated per run for clean peak-memory readings.

usage: scal_worker.py <engine> <n_rows> <threads> [width]
engines: pandas | polars | duckdb | spark | mp<P> (multiprocessing partitions)
"""
import json
import os
import sys
import time

engine = sys.argv[1]
n_rows = int(sys.argv[2])
threads = int(sys.argv[3])
width = int(sys.argv[4]) if len(sys.argv) > 4 else 4

os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["POLARS_MAX_THREADS"] = str(threads)
os.environ.setdefault("PYTHONHASHSEED", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

import synth  # noqa: E402

TUPLE_COLS = ["trust", "ts", "value_code", "offer_id"]


def widen(rel, width, tie_only=True, seed=0):
    """Pad the policy tuple to `width` coordinates (incl. value+tie)."""
    extra = max(0, width - 4)
    rng = np.random.default_rng(seed)
    cols = []
    for i in range(extra):
        c = f"q{i}"
        rel[c] = 0.0 if tie_only else rng.random(len(rel)).astype("float64")
        cols.append(c)
    return rel, ["trust", "ts"] + cols + ["value_code", "offer_id"]


def run_pandas(rel, tcols):
    t0 = time.perf_counter()
    d = rel.sort_values(["cluster_id", "field"] + tcols, kind="mergesort")
    win = d.groupby(["cluster_id", "field"], sort=False, observed=True).tail(1)
    out = win[["cluster_id", "field", "value_code"]]
    dt = time.perf_counter() - t0
    return dt, len(out)


def run_polars(rel, tcols):
    import polars as pl
    d = pl.from_pandas(rel[["cluster_id", "field"] + tcols])
    t0 = time.perf_counter()
    d = d.sort(["cluster_id", "field"] + tcols)
    win = d.group_by(["cluster_id", "field"], maintain_order=True).last()
    out = win.select(["cluster_id", "field", "value_code"])
    n = out.height
    dt = time.perf_counter() - t0
    return dt, n


def run_duckdb(rel, tcols):
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    fields = ", ".join(f"{c} := {c}" for c in tcols)
    q = (f"SELECT cluster_id, field, max(struct_pack({fields})) AS w "
         f"FROM rel GROUP BY cluster_id, field")
    t0 = time.perf_counter()
    out = con.execute(q).fetch_arrow_table()
    n = out.num_rows
    dt = time.perf_counter() - t0
    con.close()
    return dt, n


def run_spark(rel, tcols):
    java = None
    env_root = os.path.dirname(os.path.dirname(sys.executable))
    for cand in (os.path.join(sys.prefix, "Library", "lib", "jvm"),
                 os.path.join(sys.prefix, "Library")):
        for root, dirs, files in os.walk(cand):
            if "java.exe" in files and root.endswith("bin"):
                java = os.path.dirname(root)
                break
        if java:
            break
    if java:
        os.environ["JAVA_HOME"] = java
        os.environ["PATH"] = os.path.join(java, "bin") + os.pathsep + os.environ["PATH"]
    from pyspark.sql import SparkSession, functions as F
    spark = (SparkSession.builder.master(f"local[{threads}]")
             .config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", str(threads))
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .appName("tf").getOrCreate())
    sdf = spark.createDataFrame(rel[["cluster_id", "field"] + tcols])
    sdf.cache().count()
    t0 = time.perf_counter()
    win = (sdf.groupBy("cluster_id", "field")
           .agg(F.max(F.struct(*tcols)).alias("w")))
    n = win.count()
    dt = time.perf_counter() - t0
    spark.stop()
    return dt, n


def _mp_part(args):
    import pandas as _pd
    part, tcols = args
    t0 = time.perf_counter()
    d = part.sort_values(["cluster_id", "field"] + tcols, kind="mergesort")
    win = d.groupby(["cluster_id", "field"], sort=False, observed=True).tail(1)
    return win, time.perf_counter() - t0


def run_mp(rel, tcols, n_parts):
    """Partition by cluster hash; per-partition sort plan; merge partials.
    Reports wall time, critical (max) partition time, and merge time."""
    import multiprocessing as mp
    parts_idx = (rel["cluster_id"].to_numpy() % n_parts)
    parts = [rel.iloc[parts_idx == p] for p in range(n_parts)]
    t0 = time.perf_counter()
    if n_parts == 1:
        win, pt = _mp_part((parts[0], tcols))
        wall = time.perf_counter() - t0
        return {"wall_s": wall, "critical_s": pt, "merge_s": 0.0, "out_rows": len(win)}
    with mp.Pool(processes=min(n_parts, 16)) as pool:
        rets = pool.map(_mp_part, [(p, tcols) for p in parts])
    crit = max(r[1] for r in rets)
    tm = time.perf_counter()
    partials = pd.concat([r[0] for r in rets], ignore_index=True)
    d = partials.sort_values(["cluster_id", "field"] + tcols, kind="mergesort")
    win = d.groupby(["cluster_id", "field"], sort=False, observed=True).tail(1)
    merge_s = time.perf_counter() - tm
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "critical_s": crit, "merge_s": merge_s, "out_rows": len(win)}


def main():
    base = synth.load_scal_base()
    rel = synth.replicate(base, n_rows, seed=0)
    rel, tcols = widen(rel, width, tie_only=False, seed=0)
    rec = {"engine": engine, "n_rows": n_rows, "threads": threads, "width": width}
    if engine == "pandas":
        dt, n = run_pandas(rel, tcols)
    elif engine == "polars":
        dt, n = run_polars(rel, tcols)
    elif engine == "duckdb":
        dt, n = run_duckdb(rel, tcols)
    elif engine == "spark":
        dt, n = run_spark(rel, tcols)
    elif engine.startswith("mp"):
        out = run_mp(rel, tcols, int(engine[2:]))
        rec.update(out)
        dt, n = out["wall_s"], out["out_rows"]
    else:
        raise SystemExit(f"unknown engine {engine}")
    rec["time_s"] = round(dt, 4)
    rec["throughput_mrows_s"] = round(n_rows / dt / 1e6, 4)
    rec["out_rows"] = int(n)
    mi = psutil.Process().memory_info()
    rec["peak_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
