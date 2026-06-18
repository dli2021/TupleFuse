"""Experiment 12: cross-dataset portability.

MusicBrainz 20K (5 real sources): full quality protocol (silver by agreement,
trust split, all main methods) + heterogeneity + determinism mismatches.
MusicBrainz 200K: operator runtime at scale on real data.
Magellan two-source suites: coverage, exact two-source field agreement,
operator runtime, determinism mismatches."""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv  # noqa: E402
import protocol as pr  # noqa: E402
import trust as tr  # noqa: E402
import tuplefuse as tf  # noqa: E402
import baselines as bl  # noqa: E402

LP = os.path.join(DATA, "leipzig")
ZM = os.path.join(DATA, "zenodo_magellan")


def musicbrainz_quality():
    rel = pd.read_parquet(os.path.join(LP, "musicbrainz_relation.parquet"))
    rel = rel.rename(columns={"rid": "record_id"})
    collapsed, silver, _ = pr.prepare(rel, k_gold=5)
    print("MB silver pairs:", len(silver), "clusters:", silver.cluster_id.nunique(),
          flush=True)
    rows = []
    n_trust = min(300, silver["cluster_id"].nunique() // 2)
    for seed in (0, 1, 2):
        trust_c, test_c = pr.split_clusters(silver, n_trust, seed)
        res = pr.run_methods(collapsed, silver, trust_c, test_c,
                             include=["Recency-First", "Source-Priority",
                                      "Cluster-Majority", "Weighted-Voting-Global",
                                      "Weighted-Voting-Field", "EM-Truth-Discovery",
                                      "TupleFuse-Vote-Mean", "TupleFuse-Vote-LCB",
                                      "TupleFuse-Vote-LCB-Decay"])
        for m, r in res.items():
            rows.append({"dataset": "musicbrainz-20k", "seed": seed, "method": m,
                         "F1": r["F1"], "EM": r["EM"]})
    df = pd.DataFrame(rows)
    save_csv(df.round(4), "exp12_musicbrainz_per_seed.csv")
    agg = df.groupby("method")[["F1", "EM"]].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    save_csv(agg.reset_index(), "exp12_musicbrainz_summary.csv")
    print(agg.to_string(), flush=True)

    # heterogeneity on real sources
    adj = tr.adjudications(collapsed, silver)
    ttab = tr.trust_table(tr.counts_from_adjudications(adj), "mean")
    heat = ttab.pivot(index="source_id", columns="field", values="trust").round(4)
    save_csv(heat.reset_index(), "exp12_musicbrainz_heterogeneity.csv")
    print(heat.to_string(), flush=True)


def musicbrainz200_runtime():
    p = os.path.join(LP, "musicbrainz200_relation.parquet")
    if not os.path.exists(p):
        return
    rel = pd.read_parquet(p)
    d = rel.copy()
    d["trust"] = 1.0
    t0 = time.perf_counter()
    out = tf.vote_pandas(d)
    dt = time.perf_counter() - t0
    save_csv(pd.DataFrame([{"dataset": "musicbrainz-200k", "rows": len(rel),
                            "clusters": rel.cluster_id.nunique(),
                            "vote_time_s": round(dt, 3),
                            "out_rows": len(out)}]),
             "exp12_musicbrainz200_runtime.csv")
    print("MB200k rows", len(rel), "vote_time", dt, flush=True)


def determinism_mismatches(rel, n=(4, 16), seeds=3):
    d = rel.copy()
    if "trust" not in d.columns:
        d["trust"] = 1.0
    ref = tf.rank_pandas_sort(d, tuple_cols=["trust", "ts", "value", "offer_id"])
    h = tf.canonical_hash(ref)
    mism = 0
    for P in n:
        for s in range(seeds):
            rng = np.random.default_rng(31 * P + s)
            parts = tf.random_partition(d, P, rng)
            partials = [tf.rank_partial_rows(p, ["trust", "ts", "value", "offer_id"],
                                             ["cluster_id", "field"], [])
                        for p in parts if len(p)]
            merged = tf.random_merge_tree(partials, ["trust", "ts", "value", "offer_id"],
                                          ["cluster_id", "field"], [], rng)
            mism += int(tf.canonical_hash(merged[["cluster_id", "field", "value"]]) != h)
    return mism


def magellan_suite():
    rows = []
    for p in sorted(glob.glob(os.path.join(ZM, "*_relation.parquet"))):
        name = os.path.basename(p).replace("_relation.parquet", "")
        rel = pd.read_parquet(p)
        n_cl = rel["cluster_id"].nunique()
        coll = tr.collapse_per_source(rel.assign(
            offer_id=rel["offer_id"], ts=rel["ts"]))
        # two-source exact agreement per field, on clusters where both offer
        piv = coll.pivot_table(index=["cluster_id", "field"], columns="source_id",
                               values="value", aggfunc="first")
        if piv.shape[1] >= 2:
            both = piv.dropna()
            agree = float((both.iloc[:, 0] == both.iloc[:, 1]).mean()) if len(both) else np.nan
        else:
            agree = np.nan
        d = rel.copy()
        d["trust"] = 1.0
        t0 = time.perf_counter()
        out = tf.vote_pandas(d)
        dt = time.perf_counter() - t0
        cov = len(out) / max(n_cl * rel["field"].nunique(), 1)
        mism = determinism_mismatches(rel.sample(min(len(rel), 200_000),
                                                 random_state=0))
        rows.append({"dataset": name, "rows": len(rel), "clusters": n_cl,
                     "fields": rel["field"].nunique(),
                     "two_source_agreement": round(agree, 4) if agree == agree else None,
                     "vote_time_s": round(dt, 3),
                     "canonical_rows": len(out),
                     "coverage": round(cov, 4),
                     "determinism_mismatches": mism})
        print(rows[-1], flush=True)
    save_csv(pd.DataFrame(rows), "exp12_magellan.csv")


if __name__ == "__main__":
    musicbrainz_quality()
    musicbrainz200_runtime()
    magellan_suite()
