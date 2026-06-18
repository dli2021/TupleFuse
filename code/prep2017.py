"""Build the WDC Products-2017 fusion relation from HF pair files.

Pools offers from both pair sides across the four categories, dedupes by
offer id, keeps clusters with >= 2 distinct (proxy) sources, canonicalizes
title/brand/description, and derives:
  source proxy : src_(offer_id // 2**20)   (crawl-partition id bucket)
  ts proxy     : global percentile rank of offer id in [0, 1]
Output: parquet at data/wdc_products_2017/fusion_relation.parquet
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, FIELDS_2017, save_csv  # noqa: E402
from textnorm import norm_text  # noqa: E402

D2017 = os.path.join(DATA, "wdc_products_2017")
CATS = ["cameras", "computers", "shoes", "watches"]
SPLITS = ["train_xlarge", "valid_xlarge", "test"]
ATTRS = ["title", "brand", "description"]


def load_offers():
    rows = {}
    for cat in CATS:
        for sp in SPLITS:
            p = os.path.join(D2017, f"{cat}__{sp}.json.gz")
            if not os.path.exists(p):
                continue
            with gzip.open(p, "rt", encoding="utf8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for side in ("left", "right"):
                        oid = o.get(f"id_{side}")
                        if oid is None or oid in rows:
                            continue
                        rows[oid] = {
                            "offer_id": int(oid),
                            "cluster_id": int(o.get(f"cluster_id_{side}", -1)),
                            "category": cat,
                            **{a: o.get(f"{a}_{side}") for a in ATTRS},
                        }
    return pd.DataFrame(list(rows.values()))


def main():
    offers = load_offers()
    print("raw pooled offers:", len(offers), flush=True)
    offers = offers[offers["cluster_id"] >= 0].copy()

    offers["source_id"] = "src_" + (offers["offer_id"] // (1 << 20)).astype(str)
    # Pseudo-timestamp by multiplicative hashing of the offer id. The hash
    # decorrelates the time proxy from the id-bucket source proxy: a raw
    # id-rank timestamp would assign each pseudo-temporal window a disjoint
    # range of id buckets, confounding source identity with arrival time.
    h = (offers["offer_id"].astype("uint64") * np.uint64(2654435761)) \
        % np.uint64(2**32)
    offers["ts"] = (h.astype("float64") / float(2**32))

    # long format: one row per (offer, field) with a usable canonical value
    longs = []
    for a in ATTRS:
        d = offers[["cluster_id", "offer_id", "source_id", "ts", "category", a]].copy()
        d["field"] = a
        d["value"] = d[a].map(norm_text)
        longs.append(d.drop(columns=[a]))
    rel = pd.concat(longs, ignore_index=True)
    rel = rel[rel["value"].notna()].copy()

    # keep clusters that have >= 2 distinct sources overall
    ns = rel.groupby("cluster_id")["source_id"].nunique()
    keep = ns[ns >= 2].index
    rel = rel[rel["cluster_id"].isin(keep)].copy()

    out = os.path.join(D2017, "fusion_relation.parquet")
    rel.to_parquet(out, index=False)

    stats = {
        "offers": int(rel["offer_id"].nunique()),
        "clusters": int(rel["cluster_id"].nunique()),
        "sources": int(rel["source_id"].nunique()),
        "rows": int(len(rel)),
        "avg_sources_per_cluster": float(
            rel.groupby("cluster_id")["source_id"].nunique().mean()),
        "median_offers_per_cluster": float(
            rel.groupby("cluster_id")["offer_id"].nunique().median()),
        "per_field_rows": rel.groupby("field").size().to_dict(),
        "per_category_clusters": rel.groupby("category")["cluster_id"].nunique().to_dict(),
    }
    with open(os.path.join(D2017, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
