"""Experiment 3: source-field reliability heterogeneity (heatmap + variance)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv  # noqa: E402
import protocol as pr  # noqa: E402
import trust as tr  # noqa: E402


def main():
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    collapsed, silver, _ = pr.prepare(rel)
    adj = tr.adjudications(collapsed, silver)
    counts = tr.counts_from_adjudications(adj)
    ttab = tr.trust_table(counts, "mean")

    heat = ttab.pivot(index="source_id", columns="field", values="trust").round(4)
    n = counts.assign(n=counts["succ"] + counts["fail"]).pivot_table(
        index="source_id", columns="field", values="n")
    heat = heat.reindex(sorted(heat.index, key=lambda s: int(str(s).split("_")[1])))
    save_csv(heat.reset_index(), "exp03_source_field_posterior_mean.csv")
    save_csv(n.reset_index(), "exp03_source_field_n.csv")

    t = ttab.pivot(index="source_id", columns="field", values="trust")
    cover = counts.groupby("source_id")[["succ", "fail"]].sum().sum(axis=1)
    het = pd.DataFrame({
        "source_id": t.index,
        "evidence": cover.reindex(t.index).fillna(0).values,
        **{f"trust_{c}": t[c].values for c in t.columns},
        "variance": t.var(axis=1, ddof=0).values,
        "range": (t.max(axis=1) - t.min(axis=1)).values,
    }).sort_values("variance", ascending=False)
    save_csv(het.round(4), "exp03_heterogeneity_table.csv")

    glob = adj.groupby("source_id")["X"].mean()
    spread = float(het["range"].mean())
    print("mean field-range of source reliability:", round(spread, 4), flush=True)
    print(het.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
