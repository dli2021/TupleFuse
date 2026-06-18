"""Fusion baselines: recency, source priority, majority, weighted voting,
frequentist field weights, iterative truth discovery (EM-lite), oracle."""
import numpy as np
import pandas as pd
import tuplefuse as tf
import trust as tr


def recency_first(collapsed):
    d = collapsed.copy()
    d = d.sort_values(["cluster_id", "field", "ts", "offer_id", "value"], kind="mergesort")
    win = d.groupby(["cluster_id", "field"], observed=True).tail(1)
    return win[["cluster_id", "field", "value"]].reset_index(drop=True)


def source_priority(collapsed, train_adj):
    """Fixed global hierarchy: overall hit-rate per source on the training
    adjudications (all fields pooled), then rank-first selection."""
    hr = train_adj.groupby("source_id", observed=True)["X"].agg(["sum", "size"]).reset_index()
    hr["rate"] = (hr["sum"] + 1.0) / (hr["size"] + 2.0)
    pr = dict(zip(hr["source_id"], hr["rate"]))
    d = collapsed.copy()
    d["trust"] = d["source_id"].map(pr).fillna(0.5)
    return tf.rank_pandas_sort(d, tuple_cols=["trust", "ts", "value", "offer_id"])


def cluster_majority(collapsed):
    """Unweighted vote: one vote per source; ties broken by value order."""
    d = collapsed.copy()
    d["trust"] = 1.0
    return tf.vote_pandas(d)


def weighted_voting_global(collapsed, train_adj):
    """One global weight per source (hit-rate on training silver, smoothed)."""
    hr = train_adj.groupby("source_id", observed=True)["X"].agg(["sum", "size"]).reset_index()
    hr["rate"] = (hr["sum"] + 1.0) / (hr["size"] + 2.0)
    pr = dict(zip(hr["source_id"], hr["rate"]))
    d = collapsed.copy()
    d["trust"] = d["source_id"].map(pr).fillna(0.5)
    return tf.vote_pandas(d)


def weighted_voting_field(collapsed, train_adj):
    """Frequentist per-(source, field) hit rate (no Bayesian smoothing beyond
    add-one), as the non-Bayesian field-level ablation."""
    hr = (train_adj.groupby(["source_id", "field"], observed=True)["X"]
          .agg(["sum", "size"]).reset_index())
    hr["trust"] = hr["sum"] / hr["size"].clip(lower=1)
    d = collapsed.merge(hr[["source_id", "field", "trust"]],
                        on=["source_id", "field"], how="left")
    d["trust"] = d["trust"].fillna(0.5)
    return tf.vote_pandas(d)


def em_truth_discovery(collapsed, n_iter=10, init=0.8):
    """Lightweight iterative truth discovery (TruthFinder-style alternation):
    alternate weighted vote (truth step) and per-source accuracy (weight step)."""
    w = {s: init for s in collapsed["source_id"].unique()}
    d = collapsed.copy()
    truth = None
    for _ in range(n_iter):
        d["trust"] = d["source_id"].map(w)
        truth = tf.vote_pandas(d)
        j = d.merge(truth.rename(columns={"value": "tv"}), on=["cluster_id", "field"])
        j["ok"] = (j["value"] == j["tv"]).astype(float)
        acc = j.groupby("source_id", observed=True)["ok"].agg(["sum", "size"]).reset_index()
        acc["rate"] = (acc["sum"] + 1.0) / (acc["size"] + 2.0)
        w = dict(zip(acc["source_id"], acc["rate"]))
    return truth, n_iter


def oracle_upper_bound(collapsed, test_adj):
    """Per-(source, field) weights estimated on TEST silver: upper bound only."""
    hr = (test_adj.groupby(["source_id", "field"], observed=True)["X"]
          .agg(["sum", "size"]).reset_index())
    hr["trust"] = (hr["sum"] + 1.0) / (hr["size"] + 2.0)
    d = collapsed.merge(hr[["source_id", "field", "trust"]],
                        on=["source_id", "field"], how="left")
    d["trust"] = d["trust"].fillna(0.5)
    return tf.vote_pandas(d)


def tuplefuse_rank(collapsed, ttab):
    d = tr.join_trust(collapsed, ttab)
    return tf.rank_pandas_sort(d, tuple_cols=["trust", "ts", "value", "offer_id"])


def tuplefuse_vote(collapsed, ttab):
    d = tr.join_trust(collapsed, ttab)
    return tf.vote_pandas(d)
