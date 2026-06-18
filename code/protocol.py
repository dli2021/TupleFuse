"""Shared evaluation protocol for quality experiments on a fusion relation."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as bl  # noqa: E402
import silver as sv  # noqa: E402
import trust as tr  # noqa: E402
import tuplefuse as tf  # noqa: E402
from common import macro_prf, entity_exact_match  # noqa: E402


def prepare(rel: pd.DataFrame, k_gold=5):
    """Collapse to one candidate per (cluster, field, source) and build silver."""
    collapsed = tr.collapse_per_source(rel)
    silver, panel = sv.build_silver(collapsed, k=k_gold)
    return collapsed, silver, panel


def split_clusters(silver: pd.DataFrame, n_trust: int, seed: int):
    """Cluster-level split: n_trust clusters for trust estimation, rest test."""
    rng = np.random.default_rng(seed)
    clusters = np.sort(silver["cluster_id"].unique())
    rng.shuffle(clusters)
    trust_c = set(clusters[:n_trust].tolist())
    test_c = set(clusters[n_trust:].tolist())
    return trust_c, test_c


def evaluate(pred: pd.DataFrame, silver_test: pd.DataFrame):
    j = silver_test.merge(pred.rename(columns={"value": "pred"}),
                          on=["cluster_id", "field"], how="left")
    res = macro_prf(j["value"], j["pred"], j["field"])
    em = entity_exact_match(pred.rename(columns={"value": "value"}),
                            silver_test)
    res["EM"] = em
    return res


def run_methods(collapsed, silver, trust_c, test_c, delta=0.05, lam=3.0,
                include=None, oracle=False, em_iters=10):
    """Run all configured methods; returns {name: metrics dict}."""
    silver_train = silver[silver["cluster_id"].isin(trust_c)]
    silver_test = silver[silver["cluster_id"].isin(test_c)]
    test_collapsed = collapsed[collapsed["cluster_id"].isin(test_c)]

    train_adj = tr.adjudications(collapsed, silver_train)
    counts = tr.counts_from_adjudications(train_adj)

    # decayed counts: discount by time distance to the end of the train window
    t_now = float(collapsed["ts"].max())
    w = tr.decay_weights(train_adj, t_now, lam)
    counts_decay = tr.counts_from_adjudications(train_adj, weights=w)

    methods = {}
    methods["Recency-First"] = lambda: bl.recency_first(test_collapsed)
    methods["Source-Priority"] = lambda: bl.source_priority(test_collapsed, train_adj)
    methods["Cluster-Majority"] = lambda: bl.cluster_majority(test_collapsed)
    methods["Weighted-Voting-Global"] = lambda: bl.weighted_voting_global(test_collapsed, train_adj)
    methods["Weighted-Voting-Field"] = lambda: bl.weighted_voting_field(test_collapsed, train_adj)
    methods["EM-Truth-Discovery"] = lambda: bl.em_truth_discovery(test_collapsed, n_iter=em_iters)[0]
    methods["TupleFuse-Rank-Mean"] = lambda: bl.tuplefuse_rank(
        test_collapsed, tr.trust_table(counts, "mean"))
    methods["TupleFuse-Rank-MAP"] = lambda: bl.tuplefuse_rank(
        test_collapsed, tr.trust_table(counts, "map"))
    methods["TupleFuse-Rank-LCB"] = lambda: bl.tuplefuse_rank(
        test_collapsed, tr.trust_table(counts, "lcb", delta))
    methods["TupleFuse-Vote-Mean"] = lambda: bl.tuplefuse_vote(
        test_collapsed, tr.trust_table(counts, "mean"))
    methods["TupleFuse-Vote-LCB"] = lambda: bl.tuplefuse_vote(
        test_collapsed, tr.trust_table(counts, "lcb", delta))
    methods["TupleFuse-Vote-Decay"] = lambda: bl.tuplefuse_vote(
        test_collapsed, tr.trust_table(counts_decay, "mean"))
    methods["TupleFuse-Vote-LCB-Decay"] = lambda: bl.tuplefuse_vote(
        test_collapsed, tr.trust_table(counts_decay, "lcb", delta))
    if oracle:
        test_adj = tr.adjudications(collapsed, silver_test)
        methods["Oracle-Source-UB"] = lambda: bl.oracle_upper_bound(test_collapsed, test_adj)

    if include is not None:
        methods = {k: v for k, v in methods.items() if k in include}

    out = {}
    for name, fn in methods.items():
        pred = fn()
        out[name] = evaluate(pred, silver_test)
    return out
