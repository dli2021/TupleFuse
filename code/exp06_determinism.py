"""Experiment 6: partition invariance and determinism under randomized
partitioning and random merge trees (rank exact; vote in fixed-point and
float64)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv  # noqa: E402
import protocol as pr  # noqa: E402
import synth  # noqa: E402
import trust as tr  # noqa: E402
import tuplefuse as tf  # noqa: E402

PARTS = [1, 2, 4, 8, 16, 32, 64]
SEEDS = list(range(10))
TUPLE_COLS = ["trust", "ts", "value", "offer_id"]


def dataset_wdc():
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    collapsed, silver, _ = pr.prepare(rel)
    adj = tr.adjudications(collapsed, silver)
    ttab = tr.trust_table(tr.counts_from_adjudications(adj), "mean")
    return tr.join_trust(rel, ttab)


def dataset_synth(n=1_000_000):
    base = synth.load_scal_base()
    d = synth.replicate(base, n, seed=7)
    return d.rename(columns={"value_code": "value"})


def run_one(rel, name):
    ref_rank = tf.rank_pandas_sort(rel, tuple_cols=TUPLE_COLS)
    h_rank_ref = tf.canonical_hash(ref_rank)
    ref_vote = tf.vote_pandas(rel, fixed_point=True)
    h_vote_ref = tf.canonical_hash(ref_vote)

    rows = []
    for P in PARTS:
        hashes_rank, hashes_vfix, hashes_vflt = set(), set(), set()
        mism_rank = mism_vfix = mism_vflt = 0
        for s in SEEDS:
            rng = np.random.default_rng(1000 * P + s)
            parts = tf.random_partition(rel, P, rng)
            partials = [tf.rank_partial_rows(p, TUPLE_COLS, ["cluster_id", "field"], [])
                        for p in parts if len(p)]
            merged = tf.random_merge_tree(partials, TUPLE_COLS,
                                          ["cluster_id", "field"], [], rng)
            out = merged[["cluster_id", "field", "value"]]
            h = tf.canonical_hash(out)
            hashes_rank.add(h)
            mism_rank += int(h != h_rank_ref)

            vp = tf.vote_partials(rel, P, rng, fixed_point=True)
            vout = tf.vote_merge_finalize(vp, rng)
            h2 = tf.canonical_hash(vout)
            hashes_vfix.add(h2)
            mism_vfix += int(h2 != h_vote_ref)

            vpf = tf.vote_partials(rel, P, rng, fixed_point=False)
            voutf = tf.vote_merge_finalize(vpf, rng)
            h3 = tf.canonical_hash(voutf)
            hashes_vflt.add(h3)
            mism_vflt += int(h3 != h_vote_ref)

        rows.append({"dataset": name, "rows": len(rel), "partitions": P,
                     "seeds": len(SEEDS),
                     "rank_mismatches": mism_rank,
                     "rank_unique_hashes": len(hashes_rank | {h_rank_ref}),
                     "votefix_mismatches": mism_vfix,
                     "votefix_unique_hashes": len(hashes_vfix | {h_vote_ref}),
                     "votefloat_mismatches": mism_vflt,
                     "votefloat_unique_hashes": len(hashes_vflt | {h_vote_ref})})
        print(name, P, rows[-1], flush=True)
    return rows


def main():
    rows = []
    rows += run_one(dataset_wdc(), "wdc2017")
    rows += run_one(dataset_synth(1_000_000), "synthetic-1M")
    mb = os.path.join(DATA, "leipzig", "musicbrainz_relation.parquet")
    if os.path.exists(mb):
        rel = pd.read_parquet(mb)
        rel = rel.assign(trust=1.0) if "trust" not in rel.columns else rel
        rows += run_one(rel, "musicbrainz-20k")
    df = pd.DataFrame(rows)
    save_csv(df, "exp06_determinism.csv")
    tot = df[["rank_mismatches", "votefix_mismatches", "votefloat_mismatches"]].sum()
    print("TOTAL MISMATCHES:", tot.to_dict(), flush=True)


if __name__ == "__main__":
    main()
