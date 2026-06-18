"""Extract and prepare the Zenodo Magellan clean-clean ER datasets as
two-source fusion relations (clusters = connected components of gold matches)."""
import os
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA  # noqa: E402
from textnorm import norm_text  # noqa: E402

ZM = os.path.join(DATA, "zenodo_magellan")
EX = os.path.join(ZM, "extracted")


def extract():
    os.makedirs(EX, exist_ok=True)
    for t in ("magellanExistingDatasets.tar.gz", "magellanNewDatasets.tar.gz"):
        p = os.path.join(ZM, t)
        if os.path.exists(p):
            try:
                with tarfile.open(p) as tf:
                    tf.extractall(EX, filter="data")
                print("extracted", t, flush=True)
            except Exception as e:
                print("extract failed", t, e, flush=True)


def find_dataset_dirs():
    """Locate dirs containing tableA.csv/tableB.csv with a gold matches file."""
    out = {}
    for root, dirs, files in os.walk(EX):
        fl = {f.lower() for f in files}
        if "tablea.csv" in fl and "tableb.csv" in fl:
            gold = None
            for cand in ("matches.csv", "gold.csv", "perfectmapping.csv",
                         "train.csv", "test.csv"):
                if cand in fl:
                    gold = cand
                    break
            out[os.path.basename(root)] = (root, gold)
    return out


def union_find_clusters(pairs):
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)
    return {x: find(x) for x in parent}


def build_relation(root, gold_file, name):
    def read(fn):
        return pd.read_csv(os.path.join(root, fn), dtype=str,
                           keep_default_na=False, na_values=[""],
                           encoding="utf-8", encoding_errors="replace")

    A = read("tableA.csv")
    B = read("tableB.csv")
    files = {f.lower(): f for f in os.listdir(root)}
    # pool every split that carries labelled pairs
    gs = []
    for fn in ("matches.csv", "gold.csv", "perfectmapping.csv",
               "train.csv", "valid.csv", "test.csv"):
        if fn in files:
            gs.append(read(files[fn]))
    g = pd.concat(gs, ignore_index=True)
    cols = {c.lower(): c for c in g.columns}
    lid = (cols.get("ltable_id") or cols.get("table1.id") or cols.get("lid")
           or cols.get("id1") or cols.get("idtablea"))
    rid = (cols.get("rtable_id") or cols.get("table2.id") or cols.get("rid")
           or cols.get("id2") or cols.get("idtableb"))
    if lid is None or rid is None:
        print("  cannot find id columns for", name, list(g.columns)[:6], flush=True)
        return None
    lab = cols.get("label")
    if lab is not None:
        g = g[g[lab].astype(str).isin(["1", "1.0", "true", "True"])]
    g = g.dropna(subset=[lid, rid]).drop_duplicates(subset=[lid, rid])
    pairs = [("A" + str(a), "B" + str(b)) for a, b in zip(g[lid], g[rid])]
    if not pairs:
        return None
    comp = union_find_clusters(pairs)
    roots = {}
    cl = {}
    for k, r in comp.items():
        if r not in roots:
            roots[r] = len(roots)
        cl[k] = roots[r]

    idA = {c.lower(): c for c in A.columns}.get("id", A.columns[0])
    idB = {c.lower(): c for c in B.columns}.get("id", B.columns[0])
    attrs = [c for c in A.columns if c != idA and c in B.columns]
    longs = []
    for tbl, idc, tag in ((A, idA, "A"), (B, idB, "B")):
        key = tag + tbl[idc].astype(str)
        keep = key.map(cl).notna()
        t = tbl[keep].copy()
        kk = key[keep]
        for a in attrs:
            d = pd.DataFrame({
                "cluster_id": kk.map(cl).astype("int64"),
                "source_id": "src" + tag,
                "field": a.lower(),
                "value": t[a].map(norm_text),
            })
            longs.append(d)
    rel = pd.concat(longs, ignore_index=True)
    rel = rel[rel["value"].notna()].copy()
    rel["offer_id"] = np.arange(len(rel), dtype="int64")
    rel["ts"] = rel["offer_id"] / max(len(rel), 1)
    out = os.path.join(ZM, f"{name}_relation.parquet")
    rel.to_parquet(out, index=False)
    return {"name": name, "rows": len(rel),
            "clusters": int(rel.cluster_id.nunique()),
            "fields": int(rel.field.nunique())}


if __name__ == "__main__":
    extract()
    found = find_dataset_dirs()
    print("found dataset dirs:", sorted(found.keys()), flush=True)
    stats = []
    for name, (root, gold) in sorted(found.items()):
        if gold is None:
            print("skip (no gold):", name, flush=True)
            continue
        try:
            s = build_relation(root, gold, name)
            if s:
                stats.append(s)
                print(s, flush=True)
        except Exception as e:
            print("FAILED", name, repr(e), flush=True)
    pd.DataFrame(stats).to_csv(os.path.join(ZM, "magellan_stats.csv"), index=False)
