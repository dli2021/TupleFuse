"""Silver canonical labels via inter-source agreement among a gold source panel."""
import numpy as np
import pandas as pd


def gold_panel(collapsed: pd.DataFrame, k=5):
    """Top-k sources per field by offer count on that field (protocol spec)."""
    cov = (collapsed.groupby(["field", "source_id"], observed=True)
           .size().reset_index(name="cov"))
    cov = cov.sort_values(["field", "cov", "source_id"], ascending=[True, False, True],
                          kind="mergesort")
    return cov.groupby("field", observed=True).head(k)[["field", "source_id"]]


def build_silver(collapsed: pd.DataFrame, k=5, min_agree=2, min_ratio=1 / 3):
    """For each (cluster, field): silver = top normalized value among gold-panel
    sources if support >= min_agree and support/(# gold sources offering a
    value) >= min_ratio; otherwise abstain (pair excluded)."""
    panel = gold_panel(collapsed, k=k)
    g = collapsed.merge(panel, on=["field", "source_id"], how="inner")
    support = (g.groupby(["cluster_id", "field", "value"], observed=True)["source_id"]
               .nunique().reset_index(name="sup"))
    offering = (g.groupby(["cluster_id", "field"], observed=True)["source_id"]
                .nunique().reset_index(name="n_gold"))
    s = support.merge(offering, on=["cluster_id", "field"])
    s = s.sort_values(["cluster_id", "field", "sup", "value"], kind="mergesort")
    top = s.groupby(["cluster_id", "field"], observed=True).tail(1)
    ok = top[(top["sup"] >= min_agree) & (top["sup"] / top["n_gold"] >= min_ratio)]
    return ok[["cluster_id", "field", "value"]].reset_index(drop=True), panel
