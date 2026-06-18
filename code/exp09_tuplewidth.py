"""Experiment 9: tuple-width sensitivity (2/4/8/16 coordinates) at 10M rows,
plus the tie-only same-output invariance check."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_csv  # noqa: E402
import synth  # noqa: E402
import tuplefuse as tf  # noqa: E402

PY = sys.executable
SRC = os.path.dirname(os.path.abspath(__file__))
WIDTHS = [2, 4, 8, 16]
N = 10_000_000


def run_worker(engine, width):
    cmd = [PY, os.path.join(SRC, "scal_worker.py"), engine, str(N), "16", str(width)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"engine": engine, "width": width, "error": (r.stderr or r.stdout)[-300:]}


def same_output_check():
    """Extra coordinates that are constant (tie-only) must not change output."""
    base = synth.load_scal_base()
    rel = synth.replicate(base, 1_000_000, seed=3).rename(columns={"value_code": "value"})
    ref = tf.rank_pandas_sort(rel, tuple_cols=["trust", "ts", "value", "offer_id"])
    h_ref = tf.canonical_hash(ref)
    out = []
    for width in WIDTHS:
        extra = max(0, width - 4)
        d = rel.copy()
        cols = []
        for i in range(extra):
            c = f"q{i}"
            d[c] = np.float64(0.0)
            cols.append(c)
        tcols = ["trust", "ts"] + cols + ["value", "offer_id"]
        got = tf.rank_pandas_sort(d, tuple_cols=tcols)
        out.append({"width": width, "same_output": tf.canonical_hash(got) == h_ref})
    return out


def main():
    synth.load_scal_base()
    rows = []
    for w in WIDTHS:
        for eng in ("pandas", "duckdb"):
            rec = run_worker(eng, w)
            rec["width"] = w
            rows.append(rec)
            print(rec, flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "exp09_tuplewidth.csv")
    chk = pd.DataFrame(same_output_check())
    save_csv(chk, "exp09_sameoutput.csv")
    print(chk.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
