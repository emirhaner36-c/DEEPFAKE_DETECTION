"""
The 140k dataset images are already cropped face images — no MTCNN needed.
This script just builds the CSV manifests directly from the directory structure.
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

RAW_BASE     = "data/raw/140k/real_vs_fake/real-vs-fake"
MANIFEST_DIR = "data/manifests"

# Cap per class per split: keeps training practical on Mac (MPS)
CAPS = {"train": 10000, "valid": 2000, "test": 2000}
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

os.makedirs(MANIFEST_DIR, exist_ok=True)

for raw_split, out_split in SPLIT_MAP.items():
    cap     = CAPS[raw_split]
    records = []
    for cls, label in [("real", 0), ("fake", 1)]:
        folder = os.path.join(RAW_BASE, raw_split, cls)
        paths  = sorted(glob.glob(os.path.join(folder, "*.jpg")))[:cap]
        for p in paths:
            records.append({"path": os.path.abspath(p), "label": label})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    out = os.path.join(MANIFEST_DIR, f"{out_split}.csv")
    df.to_csv(out, index=False)
    print(f"{out_split:6s}: {len(df):>6,} images  (real={sum(df.label==0):,}  fake={sum(df.label==1):,})  → {out}")
