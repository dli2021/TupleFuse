"""Experiment 5: controlled source-quality drift, cumulative-state design.

At each window w the pipeline maintains the canonical state fused from ALL
candidates with ts in windows 1..w (prefix), using trust learned from
adjudications revealed after windows < w (prequential). Window-w adjudications
then update the counts. WV-static freezes one global weight per source after
W3 (the last clean window). Drift types: sudden / gradual / recurring;
strengths 2/3/5 of 17 sources; affected fields {brand},{title},{brand,title};
decay rates lambda in {0.1,0.3,1,3,5}."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, save_csv  # noqa: E402
import protocol as pr  # noqa: E402
import trust as tr  # noqa: E402
import tuplefuse as tf  # noqa: E402

N_WINDOWS = 8
LAMBDAS = [0.1, 0.3, 1.0, 3.0, 5.0]
FRACS = {"2src": 2, "3src": 3, "5src": 5}
FIELDSETS = {"brand": ["brand"], "title": ["title"], "brand+title": ["brand", "title"]}
TYPES = ["sudden", "gradual", "recurring"]
PRIOR_MEAN = 0.5
CANON = ("sudden", "3src", "brand+title")


def corrupt(rel, bad_sources, fieldset, dtype, rng):
    d = rel.copy()
    w = d["window"].values
    in_bad = d["source_id"].isin(bad_sources).values & d["field"].isin(fieldset).values
    if dtype == "sudden":
        p = np.where(w >= 4, 1.0, 0.0)
    elif dtype == "gradual":
        p = np.clip((w - 3) / 5.0, 0, 1) * 0.8
    elif dtype == "recurring":
        p = np.where(np.isin(w, [4, 5, 7, 8]), 1.0, 0.0)
    else:
        raise ValueError(dtype)
    hit = in_bad & (rng.random(len(d)) < p)
    bad_vals = ("drifted " + d["cluster_id"].astype(str) + " " + d["field"].astype(str))
    d.loc[hit, "value"] = bad_vals[hit]
    return d


def trust_from_events(events, cfg, t_now):
    if not events:
        return None
    c = pd.DataFrame(events)
    lam = cfg.get("lam")
    w = np.exp(-lam * np.maximum(t_now - c["ts"].values, 0.0)) if lam is not None \
        else np.ones(len(c))
    c["w1"] = w * c["X"]
    c["w0"] = w * (1 - c["X"])
    if cfg.get("global"):
        g = c.groupby("source_id", observed=True)[["w1", "w0"]].sum().reset_index()
        g["trust"] = (2.0 + g["w1"]) / (4.0 + g["w1"] + g["w0"])
        return g[["source_id", "trust"]]
    g = c.groupby(["source_id", "field"], observed=True)[["w1", "w0"]].sum().reset_index()
    g = g.rename(columns={"w1": "succ", "w0": "fail"})
    return tr.trust_table(g, cfg["mode"], 0.05)


def fuse(coll_prefix, ttab, cfg):
    d = coll_prefix.copy()
    if ttab is None:
        d["trust"] = PRIOR_MEAN
    elif cfg.get("global"):
        d = d.merge(ttab, on="source_id", how="left")
        d["trust"] = d["trust"].fillna(PRIOR_MEAN)
    else:
        d = tr.join_trust(d, ttab)
    return tf.vote_pandas(d)


def run_scenario(rel_w, silver, dtype, n_bad, fieldset, methods, seed=0):
    rng = np.random.default_rng(seed)
    cov = rel_w.groupby("source_id")["offer_id"].size().sort_values(ascending=False)
    bad = list(cov.index[:n_bad])
    cor = corrupt(rel_w, bad, fieldset, dtype, rng)

    out_rows, post_rows = [], []
    states = {m: [] for m in methods}
    for w in range(1, N_WINDOWS + 1):
        prefix = cor[cor["window"] <= w]
        coll = tr.collapse_per_source(prefix)
        sl_w = silver.merge(coll[["cluster_id", "field"]].drop_duplicates(),
                            on=["cluster_id", "field"])
        adj_now = tr.adjudications(coll, sl_w).assign(ts=float(w))
        for m, cfg in methods.items():
            ttab = trust_from_events(states[m], cfg, t_now=w)
            pred = fuse(coll, ttab, cfg)
            j = sl_w.merge(pred.rename(columns={"value": "p"}),
                           on=["cluster_id", "field"], how="left")
            ok = (j["p"] == j["value"]) & j["p"].notna()
            prec = ok.sum() / max(int(j["p"].notna().sum()), 1)
            rec = ok.sum() / max(len(j), 1)
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
            out_rows.append({"window": w, "method": m, "F1": f1, "n": len(j)})
            if ttab is not None and not cfg.get("global"):
                tb = ttab[ttab["source_id"].isin(bad)]
                post_rows.append({"window": w, "method": m,
                                  "bad_mean_trust": float(tb["trust"].mean()) if len(tb) else np.nan})
            if not (cfg.get("freeze_after") and w > cfg["freeze_after"]):
                states[m].extend(adj_now[["source_id", "field", "X", "ts"]]
                                 .to_dict("records"))
    return pd.DataFrame(out_rows), pd.DataFrame(post_rows), bad


def main():
    rel = pd.read_parquet(os.path.join(DATA, "wdc_products_2017", "fusion_relation.parquet"))
    rel = rel.sort_values("ts", kind="mergesort").reset_index(drop=True)
    rel["window"] = (np.arange(len(rel)) * N_WINDOWS // len(rel)) + 1
    _, silver, _ = pr.prepare(rel)

    methods = {"WV-static": {"global": True, "lam": None, "freeze_after": 3}}
    methods["TF-Vote-NoDecay"] = {"mode": "mean", "lam": None}
    for lam in LAMBDAS:
        methods[f"TF-Vote-Decay-{lam}"] = {"mode": "mean", "lam": lam}

    all_f1, all_post, meta = [], [], []
    for dtype in TYPES:
        for fname, n_bad in FRACS.items():
            for fsname, fset in FIELDSETS.items():
                if (dtype, fname, fsname) == CANON:
                    ms = methods
                else:
                    ms = {k: methods[k] for k in
                          ["WV-static", "TF-Vote-NoDecay", "TF-Vote-Decay-3.0"]}
                f1, post, bad = run_scenario(rel, silver, dtype, n_bad, fset, ms)
                sc = f"{dtype}|{fname}|{fsname}"
                f1["scenario"] = sc
                post["scenario"] = sc
                all_f1.append(f1)
                all_post.append(post)
                meta.append({"scenario": sc, "bad_sources": ",".join(map(str, bad))})
                print("done scenario", sc, flush=True)

    f1 = pd.concat(all_f1, ignore_index=True)
    post = pd.concat(all_post, ignore_index=True)
    save_csv(f1.round(4), "exp05_drift_f1.csv")
    save_csv(post.round(4), "exp05_drift_badposterior.csv")
    save_csv(pd.DataFrame(meta), "exp05_drift_meta.csv")

    rows = []
    for (sc, m), g in f1.groupby(["scenario", "method"]):
        g = g.sort_values("window")
        pre = g[g.window == 3]["F1"].iloc[0]
        post_avg = g[g.window >= 4]["F1"].mean()
        final = g[g.window == N_WINDOWS]["F1"].iloc[0]
        rec_t = np.nan
        for w in range(4, N_WINDOWS + 1):
            if g[g.window == w]["F1"].iloc[0] >= 0.95 * pre:
                rec_t = w - 3
                break
        rows.append({"scenario": sc, "method": m, "pre_F1_W3": pre,
                     "post_avg_F1": post_avg, "final_F1_W8": final,
                     "recovery_windows": rec_t})
    save_csv(pd.DataFrame(rows).round(4), "exp05_drift_summary.csv")
    canon = pd.DataFrame(rows)
    canon = canon[canon.scenario == "|".join(CANON)]
    print(canon.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
