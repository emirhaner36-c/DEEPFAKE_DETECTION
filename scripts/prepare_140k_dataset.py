"""
Prepares the '140k Real and Fake Faces' Kaggle dataset
(kaggle datasets download -d xhlulu/140k-real-and-fake-faces)

Expected layout after unzipping:
    data/raw/140k/
        real_vs_fake/
            real-vs-fake/
                train/real/
                train/fake/
                valid/real/
                valid/fake/
                test/real/
                test/fake/

Outputs face-cropped images to data/processed/ and CSV manifests to data/manifests/.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
from tqdm import tqdm
from src.preprocessing import build_mtcnn, detect_and_crop

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPLITS    = ["train", "valid", "test"]
SPLIT_MAP = {"valid": "val"}
LABELS    = {"real": 0, "fake": 1}
EXTS      = {".jpg", ".jpeg", ".png"}


def process_split(split: str, raw_base: str, out_base: str, mtcnn, max_per_class: int = 5000) -> pd.DataFrame:
    split_name = SPLIT_MAP.get(split, split)
    records    = []

    for cls_name, label in LABELS.items():
        src_dir = Path(raw_base) / split / cls_name
        dst_dir = Path(out_base) / split_name / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in src_dir.iterdir() if p.suffix.lower() in EXTS]
        images = images[:max_per_class]

        for img_path in tqdm(images, desc=f"{split}/{cls_name}"):
            out_path = dst_dir / img_path.name
            success  = detect_and_crop(str(img_path), mtcnn, str(out_path))
            if success:
                records.append({"path": str(out_path), "label": label})
            else:
                logger.warning(f"No face: {img_path.name}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw/140k/real_vs_fake/real-vs-fake")
    parser.add_argument("--out_dir",     default="data/processed")
    parser.add_argument("--manifest_dir",default="data/manifests")
    parser.add_argument("--max_per_class", type=int, default=5000,
                        help="Cap images per class per split (default 5000)")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.manifest_dir, exist_ok=True)
    mtcnn = build_mtcnn(args.device)

    for split in SPLITS:
        split_name = SPLIT_MAP.get(split, split)
        df = process_split(split, args.raw_dir, args.out_dir, mtcnn, args.max_per_class)
        csv_path = os.path.join(args.manifest_dir, f"{split_name}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"{split_name}: {len(df)} images → {csv_path}")
        logger.info(f"  Real: {sum(df.label==0)} | Fake: {sum(df.label==1)}")


if __name__ == "__main__":
    main()
