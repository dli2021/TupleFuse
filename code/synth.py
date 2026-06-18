"""Synthetic workload builders: bootstrap replication and source-count grids."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA  # noqa: E402

BASE = os.path.join(DATA, "wdc_products_2017", "scal_base.parquet")


def build_scal_base():
    """Numeric-coded candidate table with trust joined (posterior mean over the
    full silver set): the base unit for bootstrap replication."""
    import protocol as pr
    import trust as tr
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    collapsed, silver, _ = pr.prepare(rel)
    adj = tr.adjudications(collapsed, silver)
    counts = tr.counts_from_adjudications(adj)
    ttab = tr.trust_table(counts, "mean")
    d = tr.join_trust(rel, ttab)
    out = pd.DataFrame({
        "cluster_id": d["cluster_id"].astype("int64"),
        "field": d["field"].astype("category").cat.codes.astype("int8"),
        "trust": d["trust"].astype("float64"),
        "ts": d["ts"].astype("float64"),
        "value_code": pd.factorize(d["value"])[0].astype("int32"),
        "offer_id": d["offer_id"].astype("int64"),
    })
    out.to_parquet(BASE, index=False)
    return out


def load_scal_base():
    if not os.path.exists(BASE):
        return build_scal_base()
    return pd.read_parquet(BASE)


def replicate(base: pd.DataFrame, n_rows: int, seed=0):
    """Tile the base table with distinct cluster/offer id offsets, truncate to
    n_rows, and shuffle row order (bootstrap-style workload replication)."""
    n0 = len(base)
    k = int(np.ceil(n_rows / n0))
    reps = []
    c_span = int(base["cluster_id"].max()) + 1
    o_span = int(base["offer_id"].max()) + 1
    for i in range(k):
        d = base.copy(deep=False)
        d = d.assign(cluster_id=d["cluster_id"] + i * c_span,
                     offer_id=d["offer_id"] + i * o_span)
        reps.append(d)
    big = pd.concat(reps, ignore_index=True).iloc[:n_rows]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(big))
    return big.iloc[idx].reset_index(drop=True)


def synth_sources(S, n_clusters=20000, fields=3, srcs_per_cf=6,
                  n_values=5, seed=0):
    """Synthetic relation with S sources of known reliability theta_{s,j}.

    Returns (relation, truth, theta) where truth holds the planted canonical
    value per (cluster, field)."""
    rng = np.random.default_rng(seed)
    theta = rng.beta(8.0, 2.0, size=(S, fields))
    rows_c, rows_f, rows_s, rows_v = [], [], [], []
    for j in range(fields):
        src = rng.integers(0, S, size=(n_clusters, srcs_per_cf))
        th = theta[src, j]
        correct = rng.random((n_clusters, srcs_per_cf)) < th
        wrong = rng.integers(1, n_values, size=(n_clusters, srcs_per_cf))
        val = np.where(correct, 0, wrong)  # 0 = planted truth
        c = np.repeat(np.arange(n_clusters), srcs_per_cf)
        rows_c.append(c)
        rows_f.append(np.full(c.shape, j, dtype="int8"))
        rows_s.append(src.ravel())
        rows_v.append(val.ravel())
    rel = pd.DataFrame({
        "cluster_id": np.concatenate(rows_c).astype("int64"),
        "field": np.concatenate(rows_f),
        "source_id": np.concatenate(rows_s).astype("int32"),
        "value": np.concatenate(rows_v).astype("int32"),
    })
    rel["ts"] = rng.random(len(rel))
    rel["offer_id"] = np.arange(len(rel), dtype="int64")
    truth = (rel[["cluster_id", "field"]].drop_duplicates().reset_index(drop=True))
    truth["value"] = 0
    return rel, truth, theta
