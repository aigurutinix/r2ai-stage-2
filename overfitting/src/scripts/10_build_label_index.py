"""Step 4: embed the row-label vocabulary with BGE-M3 (one-off GPU job).

The index lives INSIDE the store (`artifacts/store/label_index/`) so it is
copied and SHA-256 fingerprinted by scripts/04_make_kaggle_payload.py without
any manifest change.

Where to run:
  * Kaggle GPU  (recommended, ~10-20 min): see kaggle/vifinqa-embed.ipynb, then
    download label_index/ into artifacts/store/label_index/
  * Local CPU   (slow but works overnight): python scripts/10_build_label_index.py

  python scripts/10_build_label_index.py --max-labels 0        # full vocabulary
  python scripts/10_build_label_index.py --dry-run             # just count labels
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.extraction.build_store import Store
from vifinqa.retrieval.dense import LabelEncoder, DEFAULT_MODEL, collect_labels
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--out-dir", default="", help="default: <store-dir>/label_index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-labels", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store_dir = Path(args.store_dir)
    out_dir = Path(args.out_dir) if args.out_dir else store_dir / "label_index"
    store = Store(store_dir, cache_size=2)

    print("collecting distinct row labels...")
    labels = collect_labels(store)
    if args.max_labels:
        labels = labels[:args.max_labels]
    print(f"vocabulary: {len(labels)} distinct labels")
    if args.dry_run:
        for l in labels[:10]:
            print("   ", l[:90])
        return

    enc = LabelEncoder(args.model, cache_dir=None, device=args.device,
                       batch_size=args.batch_size)
    enc.build_cache(labels, out_dir)
    print(f"done -> {out_dir}")
    print("payload: rerun scripts/04_make_kaggle_payload.py so Kaggle gets it")


if __name__ == "__main__":
    main()
