"""Experiment 2: adjudication budget. Fixed held-out test clusters; trust
budgets B sampled from the remaining pool; 10 seeds per budget."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv  # noqa: E402
import protocol as pr  # noqa: E402
import trust as tr  # noqa: E402
import baselines as bl  # noqa: E402

BUDGETS = [25, 50, 100, 200, 300]
SEEDS = list(range(10))


def main():
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    collapsed, silver, _ = pr.prepare(rel)
    clusters = np.sort(silver["cluster_id"].unique())
    rng = np.random.default_rng(123)
    rng.shuffle(clusters)
    n_test = len(clusters) - max(BUDGETS)  # pool of exactly max-budget clusters
    test_c = set(clusters[:n_test].tolist())
    pool = clusters[n_test:]
    silver_test = silver[silver["cluster_id"].isin(test_c)]
    test_collapsed = collapsed[collapsed["cluster_id"].isin(test_c)]

    rows = []
    for B in BUDGETS:
        for seed in SEEDS:
            r = np.random.default_rng(seed)
            chosen = set(r.choice(pool, size=B, replace=False).tolist())
            strain = silver[silver["cluster_id"].isin(chosen)]
            adj = tr.adjudications(collapsed, strain)
            counts = tr.counts_from_adjudications(adj)
            t_now = float(collapsed["ts"].max())
            w = tr.decay_weights(adj, t_now, 3.0)
            counts_d = tr.counts_from_adjudications(adj, weights=w)

            n_eff = counts["succ"] + counts["fail"]
            preds = {
                "Weighted-Voting-Global": bl.weighted_voting_global(test_collapsed, adj),
                "Weighted-Voting-Field": bl.weighted_voting_field(test_collapsed, adj),
                "TupleFuse-Vote-Mean": bl.tuplefuse_vote(test_collapsed, tr.trust_table(counts, "mean")),
                "TupleFuse-Vote-LCB": bl.tuplefuse_vote(test_collapsed, tr.trust_table(counts, "lcb", 0.05)),
                "TupleFuse-Vote-LCB-Decay": bl.tuplefuse_vote(test_collapsed, tr.trust_table(counts_d, "lcb", 0.05)),
            }
            for m, p in preds.items():
                res = pr.evaluate(p, silver_test)
                rows.append({"budget": B, "seed": seed, "method": m,
                             "F1": res["F1"], "EM": res["EM"],
                             "mean_n_eff": float(n_eff.mean())})
        print(f"budget {B} done", flush=True)

    df = pd.DataFrame(rows)
    save_csv(df, "exp02_budget_per_seed.csv")
    agg = (df.groupby(["budget", "method"])["F1"].agg(["mean", "std"])
           .round(4).reset_index())
    save_csv(agg, "exp02_budget_summary.csv")
    print(agg.pivot(index="budget", columns="method", values="mean").to_string(), flush=True)


if __name__ == "__main__":
    main()
