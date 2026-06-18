"""Download the external corpora used by the TupleFuse experiments.

Targets (all public):
  * WDC Products-2017 product-matching pair files (HuggingFace mirror)
  * Leipzig MusicBrainz 20K and 200K multi-source ER benchmarks
  * Magellan clean-clean ER datasets (Zenodo record 7233051)

Files land under <root>/data/ in the exact layout the preprocessing scripts
expect, where <root> is $MDM_ROOT or the parent of this script's folder.

Usage:
    python download_data.py             # download everything (skip existing)
    python download_data.py --wdc       # only WDC Products-2017
    python download_data.py --leipzig   # only MusicBrainz
    python download_data.py --magellan  # only Magellan
    python download_data.py --force     # re-download even if present
"""
import os
import sys
import urllib.request

ROOT = os.environ.get("MDM_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "data")

WDC_BASE = "https://huggingface.co/datasets/wdc/products-2017/resolve/main"
WDC_CATS = ["cameras", "computers", "shoes", "watches"]
WDC_SPLITS = ["train_xlarge", "valid_xlarge", "test"]

LEIPZIG_BASE = "https://dbs.uni-leipzig.de/files/datasets/saeedi"
LEIPZIG_FILES = ["musicbrainz-20-A01.csv.dapo", "musicbrainz-200-A01.csv.dapo"]

ZENODO_BASE = "https://zenodo.org/records/7233051/files"
ZENODO_FILES = ["magellanExistingDatasets.tar.gz", "magellanNewDatasets.tar.gz"]


def _progress(blocks, bsize, total):
    if total > 0:
        pct = min(100, blocks * bsize * 100 // total)
        sys.stdout.write(f"\r    {pct:3d}%")
        sys.stdout.flush()


def fetch(url, dest, force):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        print(f"[skip] {os.path.relpath(dest, ROOT)} (exists)")
        return
    print(f"[get ] {url}")
    try:
        urllib.request.urlretrieve(url, dest, _progress)
        print(f"\r    -> {os.path.relpath(dest, ROOT)} "
              f"({os.path.getsize(dest) // 1024} KB)")
    except Exception as e:
        print(f"\r    FAILED: {e}")


def get_wdc(force):
    out = os.path.join(DATA, "wdc_products_2017")
    for cat in WDC_CATS:
        for split in WDC_SPLITS:
            fetch(f"{WDC_BASE}/{cat}/{split}.json.gz",
                  os.path.join(out, f"{cat}__{split}.json.gz"), force)


def get_leipzig(force):
    out = os.path.join(DATA, "leipzig")
    for f in LEIPZIG_FILES:
        fetch(f"{LEIPZIG_BASE}/{f}", os.path.join(out, f), force)


def get_magellan(force):
    out = os.path.join(DATA, "zenodo_magellan")
    for f in ZENODO_FILES:
        fetch(f"{ZENODO_BASE}/{f}?download=1", os.path.join(out, f), force)


def main():
    args = set(sys.argv[1:])
    force = "--force" in args
    groups = {a for a in args if a != "--force"}
    do_all = not groups
    if do_all or "--wdc" in groups:
        get_wdc(force)
    if do_all or "--leipzig" in groups:
        get_leipzig(force)
    if do_all or "--magellan" in groups:
        get_magellan(force)
    print("\nDone. Data root:", DATA)


if __name__ == "__main__":
    main()
