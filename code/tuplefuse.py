"""TupleFuse operator implementations.

Fusion relation (pandas DataFrame) columns used here:
    cluster_id : int64
    field      : str or int8 code
    value      : str (canonicalized)  [quality experiments]
    value_code : int32 (dictionary code) [scalability experiments]
    source_id  : str or int16 code
    ts         : float64  (larger = more recent)
    offer_id   : int64    (deterministic tie-breaker)
    trust      : float64  (joined-in trust coordinate)

Two instantiations:
  Rank: lexicographic max over (trust, ts, value, tie) per (cluster, field).
  Vote: sum trust per (cluster, field, value), then lexicographic max over
        (total, value) per (cluster, field).
"""
import hashlib
import io
import numpy as np
import pandas as pd

RANK_KEYS = ["trust", "ts", "value", "offer_id"]


# ---------------------------------------------------------------- rank: pandas
def rank_pandas_sort(rel: pd.DataFrame, tuple_cols=None, group_cols=("cluster_id", "field"),
                     value_col="value"):
    """Sort-based physical plan: sort by policy tuple, take last per group."""
    tuple_cols = list(tuple_cols or RANK_KEYS)
    g = list(group_cols)
    d = rel.sort_values(g + tuple_cols, kind="mergesort")
    win = d.groupby(g, sort=False, observed=True).tail(1)
    return win[g + [value_col]].rename(columns={value_col: "value"}).reset_index(drop=True)


# ------------------------------------------------------------------ rank: numpy partial-state
def rank_partial(rel: pd.DataFrame, tuple_cols=None, group_cols=("cluster_id", "field")):
    """Per-partition partial aggregate: ONE winning row per group (mergeable state)."""
    return rank_pandas_sort(rel, tuple_cols, group_cols, "value") if "value" in rel.columns else None


def rank_partial_rows(rel: pd.DataFrame, tuple_cols, group_cols, carry_cols):
    """Partial state carrying full tuples so partials can be re-merged."""
    tuple_cols = list(tuple_cols)
    g = list(group_cols)
    d = rel.sort_values(g + tuple_cols, kind="mergesort")
    return d.groupby(g, sort=False, observed=True).tail(1)[g + tuple_cols + carry_cols]


def merge_partials(partials, tuple_cols, group_cols, carry_cols):
    """Binary merge of two (or more) partial-state tables: max per group again."""
    cat = pd.concat(partials, ignore_index=True)
    return rank_partial_rows(cat, tuple_cols, group_cols, carry_cols)


# ------------------------------------------------------------------ vote: pandas
def vote_pandas(rel: pd.DataFrame, weight_col="trust", group_cols=("cluster_id", "field"),
                value_col="value", fixed_point=True):
    """Two-stage decomposable aggregate: sum weights per value, then lex max.

    fixed_point=True accumulates weights as int64 (weight * 2**32) so addition
    is exact and merge-order invariant (paper Remark 3).
    """
    g = list(group_cols)
    d = rel[g + [value_col, weight_col]].copy()
    if fixed_point:
        d["_w"] = (d[weight_col] * (1 << 32)).round().astype("int64")
    else:
        d["_w"] = d[weight_col].astype("float64")
    tot = d.groupby(g + [value_col], sort=False, observed=True)["_w"].sum().reset_index()
    tot = tot.sort_values(g + ["_w", value_col], kind="mergesort")
    win = tot.groupby(g, sort=False, observed=True).tail(1)
    return win[g + [value_col]].rename(columns={value_col: "value"}).reset_index(drop=True)


# ---------------------------------------------------------------- engines
def rank_duckdb(rel: pd.DataFrame, threads=16, value_col="value_code"):
    """Hash-aggregate plan: max over STRUCT in DuckDB (streaming, O(N))."""
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    q = f"""
        SELECT cluster_id, field,
               max(struct_pack(trust := trust, ts := ts, vc := {value_col},
                               tie := offer_id)) AS win
        FROM rel GROUP BY cluster_id, field
    """
    out = con.execute(q).df()
    win = pd.DataFrame(out["win"].tolist())
    res = out[["cluster_id", "field"]].copy()
    res["value"] = win["vc"].values
    con.close()
    return res


def rank_polars(rel: pd.DataFrame, value_col="value_code"):
    """Polars sort-based plan (sort + group last)."""
    import polars as pl
    d = pl.from_pandas(rel[["cluster_id", "field", "trust", "ts", value_col, "offer_id"]])
    d = d.sort(["cluster_id", "field", "trust", "ts", value_col, "offer_id"])
    win = d.group_by(["cluster_id", "field"], maintain_order=True).last()
    return win.select(["cluster_id", "field", value_col]).rename({value_col: "value"}).to_pandas()


def rank_spark(rel: pd.DataFrame, value_col="value_code", parallelism=16):
    """Spark local[*] grouped struct-max (only if a JVM is available)."""
    from pyspark.sql import SparkSession, functions as F
    spark = (SparkSession.builder.master(f"local[{parallelism}]")
             .config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", str(parallelism))
             .appName("tuplefuse").getOrCreate())
    sdf = spark.createDataFrame(rel[["cluster_id", "field", "trust", "ts", value_col, "offer_id"]])
    win = (sdf.groupBy("cluster_id", "field")
           .agg(F.max(F.struct("trust", "ts", value_col, "offer_id")).alias("w")))
    out = win.select("cluster_id", "field", F.col(f"w.{value_col}").alias("value")).toPandas()
    spark.stop()
    return out


# ---------------------------------------------------------------- determinism harness
def random_partition(rel: pd.DataFrame, n_parts: int, rng: np.random.Generator):
    """Random row-level partition assignment (NOT by cluster): the hard case."""
    assign = rng.integers(0, n_parts, size=len(rel))
    return [rel.iloc[assign == p] for p in range(n_parts)]


def random_merge_tree(partials, tuple_cols, group_cols, carry_cols, rng):
    """Merge partial states pairwise in random order until one remains."""
    items = [p for p in partials if len(p)]
    if not items:
        return None
    while len(items) > 1:
        i, j = sorted(rng.choice(len(items), size=2, replace=False))
        b = items.pop(j)
        a = items.pop(i)
        items.append(merge_partials([a, b], tuple_cols, group_cols, carry_cols))
    return items[0]


def canonical_hash(df: pd.DataFrame, cols=("cluster_id", "field", "value")):
    d = df[list(cols)].copy()
    d = d.sort_values(list(cols), kind="mergesort").reset_index(drop=True)
    buf = io.BytesIO()
    d.to_csv(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def vote_partials(rel, n_parts, rng, weight_col="trust",
                  group_cols=("cluster_id", "field"), value_col="value", fixed_point=True):
    """Stage-1 partial sums per partition for the vote operator."""
    g = list(group_cols)
    parts = random_partition(rel, n_parts, rng)
    outs = []
    for p in parts:
        if not len(p):
            continue
        d = p[g + [value_col, weight_col]].copy()
        if fixed_point:
            d["_w"] = (d[weight_col] * (1 << 32)).round().astype("int64")
        else:
            d["_w"] = d[weight_col].astype("float64")
        outs.append(d.groupby(g + [value_col], sort=False, observed=True)["_w"].sum().reset_index())
    return outs


def vote_merge_finalize(partials, rng, group_cols=("cluster_id", "field"), value_col="value"):
    """Random-order pairwise merge of partial sums, then stage-2 lex max."""
    g = list(group_cols)
    items = [p for p in partials if len(p)]
    while len(items) > 1:
        i, j = sorted(rng.choice(len(items), size=2, replace=False))
        b = items.pop(j)
        a = items.pop(i)
        m = (pd.concat([a, b], ignore_index=True)
             .groupby(g + [value_col], sort=False, observed=True)["_w"].sum().reset_index())
        items.append(m)
    tot = items[0].sort_values(g + ["_w", value_col], kind="mergesort")
    win = tot.groupby(g, sort=False, observed=True).tail(1)
    return win[g + [value_col]].rename(columns={value_col: "value"}).reset_index(drop=True)
