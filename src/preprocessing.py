"""
Face detection and cropping pipeline using MTCNN (facenet-pytorch).

Usage:
    python -m src.preprocessing \
        --input_dir  data/raw \
        --output_dir data/processed \
        --manifest   data/manifests/all.csv
"""
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_mtcnn(device: str = "cpu") -> MTCNN:
    return MTCNN(
        image_size=380,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=False,      # keep only the most prominent face
        device=device,
    )


def detect_and_crop(
    image_path: str,
    mtcnn: MTCNN,
    output_path: str,
    target_size: int = 380,
    margin_ratio: float = 0.1,
) -> bool:
    """
    Detect a face in the image, crop it with a margin, save to output_path.
    Returns True on success, False if no face detected.
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # MTCNN returns boxes in [x1, y1, x2, y2] format
    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return False

    x1, y1, x2, y2 = boxes[0].astype(int)
    h, w = img_np.shape[:2]

    # Add margin
    margin_x = int((x2 - x1) * margin_ratio)
    margin_y = int((y2 - y1) * margin_ratio)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    face = img_np[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (target_size, target_size))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
    return True


def process_directory(
    input_dir: str,
    output_dir: str,
    label: int,
    mtcnn: MTCNN,
    target_size: int = 380,
) -> list[dict]:
    """
    Walk input_dir, detect faces, save crops to output_dir.
    Returns list of {path, label} dicts for the manifest.
    """
    records = []
    image_paths = [
        p for p in Path(input_dir).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    for img_path in tqdm(image_paths, desc=f"Processing label={label}"):
        rel_path   = img_path.relative_to(input_dir)
        out_path   = Path(output_dir) / rel_path
        out_path   = out_path.with_suffix(".jpg")

        success = detect_and_crop(
            str(img_path), mtcnn, str(out_path), target_size
        )

        if success:
            records.append({"path": str(out_path), "label": label})
        else:
            logger.warning(f"No face detected: {img_path}")

    return records


def build_manifest(
    real_dir: str,
    fake_dir: str,
    output_dir: str,
    manifest_out: str,
    device: str = "cpu",
    target_size: int = 380,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Full pipeline: detect faces from real/fake dirs, build train/val/test manifests.
    """
    mtcnn = build_mtcnn(device)

    logger.info("Processing REAL images...")
    real_records = process_directory(real_dir, os.path.join(output_dir, "real"), 0, mtcnn, target_size)

    logger.info("Processing FAKE images...")
    fake_records = process_directory(fake_dir, os.path.join(output_dir, "fake"), 1, mtcnn, target_size)

    df = pd.DataFrame(real_records + fake_records).sample(frac=1, random_state=seed).reset_index(drop=True)

    n       = len(df)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_test - n_val

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]

    os.makedirs(os.path.dirname(manifest_out) or ".", exist_ok=True)
    base = os.path.dirname(manifest_out)

    train_df.to_csv(os.path.join(base, "train.csv"), index=False)
    val_df.to_csv(os.path.join(base,   "val.csv"),   index=False)
    test_df.to_csv(os.path.join(base,  "test.csv"),  index=False)

    logger.info(f"Manifests saved to {base}/")
    logger.info(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"  Real:  {sum(df.label==0)} | Fake: {sum(df.label==1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake preprocessing pipeline")
    parser.add_argument("--real_dir",    required=True)
    parser.add_argument("--fake_dir",    required=True)
    parser.add_argument("--output_dir",  default="data/processed")
    parser.add_argument("--manifest_dir",default="data/manifests")
    parser.add_argument("--image_size",  type=int, default=380)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    build_manifest(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        output_dir=args.output_dir,
        manifest_out=os.path.join(args.manifest_dir, "all.csv"),
        device=args.device,
        target_size=args.image_size,
    )
