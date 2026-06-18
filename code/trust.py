"""Beta-Bernoulli source-field trust: estimation, summaries, decay."""
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

PRIOR_A = 2.0
PRIOR_B = 2.0


def collapse_per_source(rel: pd.DataFrame):
    """A4: one usable candidate per (cluster, field, source): modal value,
    ties broken by recency then offer id (deterministic)."""
    d = rel.copy()
    cnt = (d.groupby(["cluster_id", "field", "source_id", "value"], observed=True)
             .agg(n=("offer_id", "size"), ts=("ts", "max"), offer_id=("offer_id", "max"))
             .reset_index())
    cnt = cnt.sort_values(["cluster_id", "field", "source_id", "n", "ts", "offer_id", "value"],
                          kind="mergesort")
    win = cnt.groupby(["cluster_id", "field", "source_id"], observed=True).tail(1)
    return win[["cluster_id", "field", "source_id", "value", "ts", "offer_id"]].reset_index(drop=True)


def adjudications(collapsed: pd.DataFrame, silver: pd.DataFrame):
    """Join source candidates to silver labels: one Bernoulli outcome per
    (cluster, field, source) with X = 1{candidate == silver}."""
    j = collapsed.merge(silver.rename(columns={"value": "silver"}),
                        on=["cluster_id", "field"], how="inner")
    j["X"] = (j["value"] == j["silver"]).astype("int8")
    return j[["cluster_id", "field", "source_id", "X", "ts"]]


def counts_from_adjudications(adj: pd.DataFrame, weights=None):
    """Aggregate (possibly weighted) successes/failures per (source, field)."""
    d = adj.copy()
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype="float64")
    d["w1"] = w * d["X"]
    d["w0"] = w * (1 - d["X"])
    g = d.groupby(["source_id", "field"], observed=True)[["w1", "w0"]].sum().reset_index()
    g = g.rename(columns={"w1": "succ", "w0": "fail"})
    return g


def trust_table(counts: pd.DataFrame, mode="mean", delta=0.05,
                prior_a=PRIOR_A, prior_b=PRIOR_B):
    """Map counts to a scalar trust per (source, field).

    mode in {mean, map, lcb}.  LCB_{1-delta} = Beta posterior delta-quantile.
    """
    t = counts.copy()
    a = prior_a + t["succ"].values
    b = prior_b + t["fail"].values
    if mode == "mean":
        s = a / (a + b)
    elif mode == "map":
        s = np.where((a > 1) & (b > 1), (a - 1) / np.maximum(a + b - 2, 1e-9), a / (a + b))
    elif mode == "lcb":
        s = beta_dist.ppf(delta, a, b)
    else:
        raise ValueError(mode)
    t["trust"] = s
    t["alpha"] = a
    t["beta"] = b
    t["n_eff"] = t["succ"] + t["fail"]
    return t[["source_id", "field", "trust", "alpha", "beta", "n_eff"]]


def decay_weights(adj: pd.DataFrame, t_now: float, lam: float):
    """Exponential time-discount weights for adjudication events."""
    return np.exp(-lam * np.maximum(t_now - adj["ts"].values, 0.0))


def join_trust(rel: pd.DataFrame, ttab: pd.DataFrame, default=None,
               prior_a=PRIOR_A, prior_b=PRIOR_B):
    """Broadcast-join trust into the candidate relation; cold sources get the
    prior mean (or the supplied default)."""
    if default is None:
        default = prior_a / (prior_a + prior_b)
    out = rel.merge(ttab[["source_id", "field", "trust"]],
                    on=["source_id", "field"], how="left")
    out["trust"] = out["trust"].fillna(default)
    return out
