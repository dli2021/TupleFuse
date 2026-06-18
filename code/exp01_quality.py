"""Experiment 1: static fusion quality on WDC Products-2017 (3 seeded splits)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv, append_jsonl, env_record  # noqa: E402
import protocol as pr  # noqa: E402

SEEDS = [0, 1, 2]
N_TRUST = 300

# Methods reported on the WDC held-out silver test set (paper Table VII).
METHODS = ["Source-Priority", "Cluster-Majority", "EM-Truth-Discovery",
           "Weighted-Voting-Global", "Weighted-Voting-Field",
           "TupleFuse-Vote-Mean", "TupleFuse-Vote-LCB"]


def main():
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    collapsed, silver, panel = pr.prepare(rel)

    n_pairs = len(silver)
    n_clusters = silver["cluster_id"].nunique()
    # majority-vs-silver coincidence over the full source set
    import baselines as bl
    maj = bl.cluster_majority(collapsed)
    j = silver.merge(maj.rename(columns={"value": "m"}), on=["cluster_id", "field"], how="left")
    coincide = float((j["value"] == j["m"]).mean())
    print(f"silver pairs={n_pairs} clusters={n_clusters} majority-coincidence={coincide:.3f}",
          flush=True)
    append_jsonl("exp01_meta.jsonl", {
        "silver_pairs": n_pairs, "silver_clusters": n_clusters,
        "majority_coincidence": coincide, "env": env_record()})

    rows = []
    for seed in SEEDS:
        trust_c, test_c = pr.split_clusters(silver, N_TRUST, seed)
        res = pr.run_methods(collapsed, silver, trust_c, test_c, include=METHODS)
        for m, r in res.items():
            rows.append({"seed": seed, "method": m, "P": r["P"], "R": r["R"],
                         "F1": r["F1"], "EM": r["EM"]})
            print(f"seed {seed}  {m:28s} F1={r['F1']:.4f} EM={r['EM']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    save_csv(df, "exp01_quality_per_seed.csv")
    agg = (df.groupby("method")[["P", "R", "F1", "EM"]]
           .agg(["mean", "std"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    save_csv(agg, "exp01_quality_summary.csv")
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
