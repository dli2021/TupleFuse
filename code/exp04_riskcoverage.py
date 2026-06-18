"""Experiment 4: risk-aware LCB and abstention (risk/precision-coverage)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv  # noqa: E402
import protocol as pr  # noqa: E402
import trust as tr  # noqa: E402
import tuplefuse as tf  # noqa: E402

DELTAS = [0.20, 0.10, 0.05, 0.01]
TAUS = [0.0, 0.2, 0.4, 0.6, 0.8]
SEEDS = [0, 1, 2]
N_TRUST = 300


def vote_with_share(d):
    """Vote winner plus its normalized trust share (decision confidence)."""
    g = ["cluster_id", "field"]
    t = (d.groupby(g + ["value"], observed=True)["trust"].sum().reset_index(name="V"))
    tot = t.groupby(g, observed=True)["V"].transform("sum")
    t["share"] = t["V"] / tot.clip(lower=1e-12)
    t = t.sort_values(g + ["V", "value"], kind="mergesort")
    win = t.groupby(g, observed=True).tail(1)
    return win[g + ["value", "share"]].reset_index(drop=True)


def main():
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    collapsed, silver, _ = pr.prepare(rel)

    rows = []
    for seed in SEEDS:
        trust_c, test_c = pr.split_clusters(silver, N_TRUST, seed)
        silver_train = silver[silver["cluster_id"].isin(trust_c)]
        silver_test = silver[silver["cluster_id"].isin(test_c)]
        test_collapsed = collapsed[collapsed["cluster_id"].isin(test_c)]
        adj = tr.adjudications(collapsed, silver_train)
        counts = tr.counts_from_adjudications(adj)
        for delta in DELTAS:
            ttab = tr.trust_table(counts, "lcb", delta)
            d = tr.join_trust(test_collapsed, ttab)
            win = vote_with_share(d)
            j = silver_test.merge(win.rename(columns={"value": "pred"}),
                                  on=["cluster_id", "field"], how="left")
            for tau in TAUS:
                ans = j["pred"].notna() & (j["share"] >= tau)
                n_ans = int(ans.sum())
                correct = int(((j["pred"] == j["value"]) & ans).sum())
                prec = correct / max(n_ans, 1)
                cov = n_ans / len(j)
                rec = correct / len(j)
                f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
                rows.append({"seed": seed, "delta": delta, "tau": tau,
                             "precision": prec, "coverage": cov,
                             "recall": rec, "F1": f1,
                             "selective_risk": 1 - prec})
    df = pd.DataFrame(rows)
    save_csv(df.round(4), "exp04_riskcoverage_per_seed.csv")
    agg = (df.groupby(["delta", "tau"])[["precision", "coverage", "F1", "selective_risk"]]
           .mean().round(4).reset_index())
    save_csv(agg, "exp04_riskcoverage_summary.csv")
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
