"""Build fusion relations for the Leipzig MusicBrainz multi-source ER
benchmarks (20K and 200K; 5 real sources, DAPO format with CID gold clusters)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA  # noqa: E402
from textnorm import norm_text, norm_number  # noqa: E402

LP = os.path.join(DATA, "leipzig")
MB_FIELDS = ["title", "artist", "album", "number", "year", "length", "language"]
NUMERIC = {"year", "length", "number"}


def build_musicbrainz(name, out_name):
    p = os.path.join(LP, name)
    df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""],
                     encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip().lower() for c in df.columns]
    # DAPO columns: tid (tuple id), cid (gold cluster id), ctid, sourceid,
    # id, number, title, length, artist, album, year, language
    longs = []
    for f in MB_FIELDS:
        if f not in df.columns:
            continue
        d = df[["cid", "sourceid", "id", f]].copy()
        d.columns = ["cluster_id", "source_id", "rid", "raw"]
        d["field"] = f
        fn = norm_number if f in NUMERIC else norm_text
        d["value"] = d["raw"].map(fn)
        longs.append(d.drop(columns=["raw"]))
    rel = pd.concat(longs, ignore_index=True)
    rel = rel[rel["value"].notna()].copy()
    rel["cluster_id"] = rel["cluster_id"].astype("int64")
    rel["source_id"] = "s" + rel["source_id"].astype(str)
    rel["offer_id"] = np.arange(len(rel), dtype="int64")
    rel["ts"] = rel["offer_id"] / len(rel)
    ns = rel.groupby("cluster_id")["source_id"].nunique()
    rel = rel[rel["cluster_id"].isin(ns[ns >= 2].index)]
    out = os.path.join(LP, out_name)
    rel.to_parquet(out, index=False)
    print(out_name, "rows:", len(rel), "clusters:", rel.cluster_id.nunique(),
          "sources:", rel.source_id.nunique(), flush=True)
    return rel


if __name__ == "__main__":
    build_musicbrainz("musicbrainz-20-A01.csv.dapo", "musicbrainz_relation.parquet")
    build_musicbrainz("musicbrainz-200-A01.csv.dapo", "musicbrainz200_relation.parquet")
