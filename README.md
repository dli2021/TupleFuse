# TupleFuse: Reproduction Package

This package contains the source code and result data behind every experiment
reported in the paper *TupleFuse: Trust-Aware Entity Fusion via Associative
Relational Aggregates*. Each table and figure in the paper can be regenerated
from the scripts in `code/`, using the public datasets whose download
instructions are given below. Precomputed result files are provided in
`results/` so the paper numbers can be inspected without re-running anything.

## 1. Directory layout

```
mdm_icde19_code/
  README.md             this file
  requirements.txt      Python dependencies
  code/                 all experiment and preprocessing source
    common.py             paths, metrics (macro P/R/F1, EM), hashing, I/O
    textnorm.py           value canonicalization (NFKC, case, punctuation)
    tuplefuse.py          the TupleFuse operator (rank + vote; engine variants)
    trust.py              Beta-Bernoulli trust: estimation, summaries, decay
    silver.py             silver canonical labels by gold-panel agreement
    baselines.py          procedural / weighted / iterative baselines
    protocol.py           shared evaluation protocol (splits, scoring)
    synth.py              synthetic workloads and bootstrap replication
    scal_worker.py        isolated timing subprocess (one engine x size)
    prep2017.py           build the WDC Products-2017 fusion relation
    prep_leipzig.py       build the MusicBrainz 20K / 200K fusion relations
    prep_magellan.py      build the 13 Magellan two-source fusion relations
    exp01_quality.py      Table VII   (fusion quality on WDC silver labels)
    exp02_budget.py       Figure 4    (adjudication-budget sweep)
    exp03_heterogeneity.py Figure 3   (source-field reliability heatmap)
    exp04_riskcoverage.py Figure 5    (trust-share abstention gate)
    exp05_drift.py        Figures 6-7, Table VIII (source-quality drift)
    exp06_determinism.py  Table II    (randomized-plan determinism)
    exp07_scal_rows.py    Table III, Figure 1 (row scaling, engine portability)
    exp08_scal_partitions.py Figure 2 (single-host partition / thread scaling)
    exp09_tuplewidth.py   Table IV    (tuple-width sensitivity)
    exp10_sourcecount.py  Table V     (source-count sensitivity, synthetic)
    exp11_streaming.py    Table VI    (streaming micro-batch execution)
    exp12_portability.py  External validity (MusicBrainz, Magellan)
    download_data.py      fetch the external corpora
    run_all.py            end-to-end driver
  results/              precomputed CSV / JSONL result files (see Section 5)
  data/                 created on download; holds the external corpora
```

## 2. Environment

The reference environment is a Windows workstation with an Intel Core i9-9880H
CPU (8 physical cores, 16 hardware threads) and 16 GB of RAM, running Python
3.12. All experiments are CPU workloads.

```bash
# from the package root
python -m venv .venv                 # or: conda create -n tuplefuse python=3.12
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # Linux / macOS
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `pyarrow`, `psutil`, `polars`, and `duckdb` cover
the full set of accuracy, determinism, drift, budget, risk, source-count, and
streaming experiments. `pyspark` is optional and only needed for the Spark
local-mode points in Figure 1 (`exp07_scal_rows.py --spark`); it requires a
Java 17+ runtime on `PATH`.

Determinism: the scripts pin thread counts (`OMP/MKL/NUMEXPR_NUM_THREADS=16`),
set `PYTHONHASHSEED=0`, and use fixed numpy seeds, so re-runs reproduce the
shipped CSVs. The working root defaults to the package directory; set the
`MDM_ROOT` environment variable to relocate `data/` and `results/`.

## 3. Getting the external data

Three public sources are used. The helper script downloads all of them into
`data/` in the layout the preprocessing scripts expect:

```bash
python code/download_data.py            # everything
python code/download_data.py --wdc      # only WDC Products-2017
python code/download_data.py --leipzig  # only MusicBrainz
python code/download_data.py --magellan # only Magellan
```

The equivalent direct sources, if you prefer to download manually:

1. **WDC Products-2017** (product-matching pair files; HuggingFace mirror).
   For each category in {cameras, computers, shoes, watches} and each split in
   {train_xlarge, valid_xlarge, test}, fetch
   `https://huggingface.co/datasets/wdc/products-2017/resolve/main/<category>/<split>.json.gz`
   and save it as `data/wdc_products_2017/<category>__<split>.json.gz`.
   Dataset page: https://huggingface.co/datasets/wdc/products-2017

2. **Leipzig MusicBrainz** (5-source multi-source ER benchmark) into
   `data/leipzig/`:
   - https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-20-A01.csv.dapo
   - https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-200-A01.csv.dapo

3. **Magellan clean-clean ER datasets** (Zenodo record 7233051) into
   `data/zenodo_magellan/`:
   - https://zenodo.org/records/7233051/files/magellanExistingDatasets.tar.gz
   - https://zenodo.org/records/7233051/files/magellanNewDatasets.tar.gz

The WDC English pair files are a few hundred MB gzipped; the Magellan
`magellanExistingDatasets.tar.gz` archive is about 590 MB. The synthetic and
bootstrap-replicated workloads (Tables IV, V, and the scaling studies) are
generated on the fly from fixed seeds and need no download.

## 4. Running the experiments

After installing dependencies and downloading the data:

```bash
# preprocessing + all accuracy / determinism / drift / portability experiments
python code/run_all.py

# timing experiments (run on an otherwise idle machine)
python code/run_all.py --timing
```

You can also run any single experiment directly, for example:

```bash
python code/prep2017.py            # build the WDC fusion relation first
python code/exp01_quality.py       # then reproduce Table VII
```

Each `prepXXXX.py` must run before the experiments that consume its relation
(`run_all.py` orders this for you). The first phase completes in roughly ten to
twenty minutes on the reference machine; the timing phase scales to 50M rows
and takes longer. Every script writes its outputs into `results/` and prints a
summary table to stdout.

The dataset statistics in Table I are printed by the preprocessing scripts
(`prep2017.py`, `prep_leipzig.py`, `prep_magellan.py`) and by `synth.py`.

## 5. Result files and paper mapping

The `results/` directory holds one or more files per experiment. Re-running a
script overwrites its own files in place.

| Paper element | Script | Result file(s) |
|---|---|---|
| Table II (determinism) | exp06_determinism.py | exp06_determinism.csv |
| Table III, Figure 1 (row scaling) | exp07_scal_rows.py | exp07_scal_rows.csv |
| Figure 2 (partition / thread scaling) | exp08_scal_partitions.py | exp08_scal_partitions.csv |
| Table IV (tuple width) | exp09_tuplewidth.py | exp09_tuplewidth.csv, exp09_sameoutput.csv |
| Table V (source count) | exp10_sourcecount.py | exp10_sourcecount.csv |
| Table VI (streaming) | exp11_streaming.py | exp11_streaming.csv, exp11_streaming_summary.csv |
| Table VII (fusion quality) | exp01_quality.py | exp01_quality_summary.csv, exp01_quality_per_seed.csv, exp01_meta.jsonl |
| Figure 3 (heterogeneity) | exp03_heterogeneity.py | exp03_source_field_posterior_mean.csv, exp03_source_field_n.csv, exp03_heterogeneity_table.csv |
| Figure 4 (budget) | exp02_budget.py | exp02_budget_summary.csv, exp02_budget_per_seed.csv |
| Figure 5 (risk-coverage) | exp04_riskcoverage.py | exp04_riskcoverage_summary.csv, exp04_riskcoverage_per_seed.csv |
| Figures 6-7, Table VIII (drift) | exp05_drift.py | exp05_drift_f1.csv, exp05_drift_summary.csv, exp05_drift_badposterior.csv, exp05_drift_meta.csv |
| External validity (MusicBrainz, Magellan) | exp12_portability.py | exp12_musicbrainz_summary.csv, exp12_musicbrainz_per_seed.csv, exp12_musicbrainz_heterogeneity.csv, exp12_musicbrainz200_runtime.csv, exp12_magellan.csv |

`exp01_meta.jsonl` records the silver-label statistics (661 labelled pairs
across 477 clusters; 94.6% cluster-majority coincidence) and the run
environment. The drift `*_summary.csv` and `*_f1.csv` files contain the full
window-by-window grid; the paper reports the three-source, brand-and-title
cells described in the Figure 6 / Table VIII captions.
