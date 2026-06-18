"""Experiment 11: streaming micro-batch execution. 20 batches by timestamp;
per batch: broadcast-join trust, fuse, receive delayed adjudication for q% of
resolved labelled fields, update Beta counts in closed form."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv  # noqa: E402
import protocol as pr  # noqa: E402
import trust as tr  # noqa: E402
import tuplefuse as tf  # noqa: E402

T = 20
QS = [0.01, 0.05, 0.10, 0.20]
PRIOR_MEAN = 0.5


def main():
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    rel = rel.sort_values("ts", kind="mergesort").reset_index(drop=True)
    rel["batch"] = (np.arange(len(rel)) * T // len(rel)) + 1
    _, silver, _ = pr.prepare(rel)

    rows = []
    for q in QS:
        rng = np.random.default_rng(42)
        counts = {}  # (source, field) -> [succ, fail]
        for b in range(1, T + 1):
            rb = rel[rel["batch"] == b]
            t0 = time.perf_counter()
            coll = tr.collapse_per_source(rb)
            if counts:
                ct = pd.DataFrame([(s, f, v[0], v[1]) for (s, f), v in counts.items()],
                                  columns=["source_id", "field", "succ", "fail"])
                ttab = tr.trust_table(ct, "lcb", 0.05)
                d = tr.join_trust(coll, ttab)
            else:
                d = coll.copy()
                d["trust"] = PRIOR_MEAN
            pred = tf.vote_pandas(d)
            fuse_s = time.perf_counter() - t0

            sl_b = silver.merge(pred[["cluster_id", "field"]].drop_duplicates(),
                                on=["cluster_id", "field"])
            j = sl_b.merge(pred.rename(columns={"value": "p"}),
                           on=["cluster_id", "field"], how="left")
            ok = (j["p"] == j["value"]) & j["p"].notna()
            f1_n = len(j)
            prec = ok.sum() / max(int(j["p"].notna().sum()), 1)
            rec = ok.sum() / max(f1_n, 1)
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

            # delayed adjudication for q% of resolved labelled fields
            t1 = time.perf_counter()
            samp = sl_b.sample(frac=q, random_state=int(1000 * q) + b) if len(sl_b) else sl_b
            n_upd = 0
            if len(samp):
                adj = tr.adjudications(coll, samp)
                for s, f, x in adj[["source_id", "field", "X"]].itertuples(index=False):
                    k = (s, f)
                    if k not in counts:
                        counts[k] = [0.0, 0.0]
                    counts[k][0 if x else 1] += 1.0
                    n_upd += 1
            upd_s = time.perf_counter() - t1
            rows.append({"q": q, "batch": b, "batch_rows": len(rb),
                         "fuse_latency_s": round(fuse_s, 4),
                         "update_s": round(upd_s, 4),
                         "updates": n_upd,
                         "updates_per_s": round(n_upd / upd_s, 1) if upd_s > 0 and n_upd else 0.0,
                         "trust_pairs": len(counts),
                         "F1": round(f1, 4), "n_eval": f1_n})
        print("done q =", q, flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "exp11_streaming.csv")
    agg = (df.groupby("q").agg(mean_latency_s=("fuse_latency_s", "mean"),
                               p95_latency_s=("fuse_latency_s", lambda x: float(np.percentile(x, 95))),
                               mean_updates_per_s=("updates_per_s", "mean"),
                               final_F1=("F1", "last"),
                               first5_F1=("F1", lambda x: float(x.iloc[:5].mean())),
                               last5_F1=("F1", lambda x: float(x.iloc[-5:].mean())))
           .round(4).reset_index())
    save_csv(agg, "exp11_streaming_summary.csv")
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
