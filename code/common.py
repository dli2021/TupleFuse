"""Shared utilities: paths, threading, timing, memory, hashing, results I/O."""
import os, sys, io, json, time, hashlib, platform, datetime

# Fix thread counts and hash seed BEFORE numpy import for reproducibility.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd
import psutil

# Working root: set MDM_ROOT to relocate data/results, otherwise default to
# the parent of the directory holding this file (so the shipped layout
# <pkg>/code/common.py resolves to <pkg>/{data,results}).
ROOT = os.environ.get("MDM_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
LOG = os.path.join(ROOT, "logs")
for _d in (RES, FIG, LOG):
    os.makedirs(_d, exist_ok=True)

FIELDS_2017 = ["title", "brand", "description"]


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def peak_mb():
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 1e6


def env_record():
    rec = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import polars as pl
        rec["polars"] = pl.__version__
    except Exception:
        pass
    try:
        import duckdb
        rec["duckdb"] = duckdb.__version__
    except Exception:
        pass
    return rec


def save_csv(df: pd.DataFrame, name: str):
    p = os.path.join(RES, name)
    df.to_csv(p, index=False)
    print(f"[saved] {p}  ({len(df)} rows)", flush=True)
    return p


def append_jsonl(name: str, obj: dict):
    p = os.path.join(RES, name)
    with open(p, "a", encoding="utf8") as f:
        f.write(json.dumps(obj) + "\n")
    return p


def sha256_of_table(df: pd.DataFrame, cols):
    """Order-insensitive canonical hash: sort by cols, serialize, hash."""
    d = df[list(cols)].sort_values(list(cols), kind="mergesort").reset_index(drop=True)
    buf = io.BytesIO()
    d.to_csv(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def macro_prf(y_true: pd.Series, y_pred: pd.Series, fields: pd.Series):
    """Macro P/R/F1 over fields for hard-decision canonical values.

    A prediction is counted per labelled (cluster, field) pair. Abstentions
    (pred is None/NaN) reduce recall but not precision.
    """
    rows = []
    for f in sorted(fields.unique()):
        m = fields == f
        t, p = y_true[m], y_pred[m]
        answered = p.notna()
        correct = (p == t) & answered
        prec = correct.sum() / max(int(answered.sum()), 1)
        rec = correct.sum() / max(int(m.sum()), 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        rows.append((f, prec, rec, f1, int(m.sum()), int(answered.sum())))
    df = pd.DataFrame(rows, columns=["field", "P", "R", "F1", "n", "answered"])
    return {
        "P": float(df.P.mean()),
        "R": float(df.R.mean()),
        "F1": float(df.F1.mean()),
        "per_field": df,
    }


def entity_exact_match(pred: pd.DataFrame, truth: pd.DataFrame):
    """Share of clusters whose every labelled field is answered correctly."""
    m = truth.merge(pred, on=["cluster_id", "field"], how="left",
                    suffixes=("_t", "_p"))
    m["ok"] = (m["value_t"] == m["value_p"]) & m["value_p"].notna()
    g = m.groupby("cluster_id")["ok"].all()
    return float(g.mean())
