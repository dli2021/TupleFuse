"""Experiment 10: source-count sensitivity with planted reliabilities.
Trust table size O(|S||J|), posterior update time, fusion time, and F1 against
the planted truth as the number of sources grows."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_csv  # noqa: E402
import synth  # noqa: E402
import trust as tr  # noqa: E402
import tuplefuse as tf  # noqa: E402

S_GRID = [10, 50, 100, 500, 1000, 5000]
N_CLUSTERS = 20000
N_LABELLED = 2000


def main():
    rows = []
    for S in S_GRID:
        rel, truth, theta = synth.synth_sources(S, n_clusters=N_CLUSTERS, seed=11)
        rel["source_id"] = rel["source_id"].astype("int32")
        lab = truth[truth["cluster_id"] < N_LABELLED]
        test = truth[truth["cluster_id"] >= N_LABELLED]

        coll = rel.rename(columns={})  # already one candidate per (c,f,s)
        t0 = time.perf_counter()
        adj = tr.adjudications(coll, lab)
        counts = tr.counts_from_adjudications(adj)
        update_s = time.perf_counter() - t0

        table_bytes = int(S * 3 * 2 * 8)  # S x J x (alpha, beta) float64
        for mode in ("mean",):
            ttab = tr.trust_table(counts, mode, 0.05)
            d = tr.join_trust(rel[rel["cluster_id"] >= N_LABELLED], ttab)
            t1 = time.perf_counter()
            pred = tf.vote_pandas(d)
            fuse_s = time.perf_counter() - t1
            j = test.merge(pred.rename(columns={"value": "p"}),
                           on=["cluster_id", "field"], how="left")
            f1 = float(((j["p"] == j["value"]) & j["p"].notna()).mean())
            rows.append({"S": S, "mode": mode, "rows": len(rel),
                         "update_s": round(update_s, 3),
                         "fuse_s": round(fuse_s, 3),
                         "trust_table_bytes": table_bytes,
                         "n_adjudications": len(adj),
                         "acc_vs_planted": round(f1, 4)})
        # majority baseline
        d = rel[rel["cluster_id"] >= N_LABELLED].copy()
        d["trust"] = 1.0
        pred = tf.vote_pandas(d)
        j = test.merge(pred.rename(columns={"value": "p"}),
                       on=["cluster_id", "field"], how="left")
        rows.append({"S": S, "mode": "majority", "rows": len(rel),
                     "update_s": 0.0, "fuse_s": np.nan,
                     "trust_table_bytes": 0, "n_adjudications": 0,
                     "acc_vs_planted": round(float(((j["p"] == j["value"]) & j["p"].notna()).mean()), 4)})
        # oracle theta
        th = pd.DataFrame({"source_id": np.repeat(np.arange(S), 3),
                           "field": np.tile(np.arange(3, dtype="int8"), S),
                           "trust": theta.ravel()})
        d = rel[rel["cluster_id"] >= N_LABELLED].merge(th, on=["source_id", "field"])
        pred = tf.vote_pandas(d)
        j = test.merge(pred.rename(columns={"value": "p"}),
                       on=["cluster_id", "field"], how="left")
        rows.append({"S": S, "mode": "oracle-theta", "rows": len(rel),
                     "update_s": 0.0, "fuse_s": np.nan,
                     "trust_table_bytes": int(S * 3 * 8), "n_adjudications": 0,
                     "acc_vs_planted": round(float(((j["p"] == j["value"]) & j["p"].notna()).mean()), 4)})
        print("done S =", S, flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "exp10_sourcecount.csv")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
