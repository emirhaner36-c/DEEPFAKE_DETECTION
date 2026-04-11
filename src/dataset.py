"""
Dataset class for deepfake detection.
Expects a CSV manifest with columns: path (absolute or relative to data_dir), label (0=real, 1=fake)
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, split: str, cfg: dict) -> A.Compose:
    aug = cfg.get("augmentation", {})

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5 if aug.get("horizontal_flip", True) else 0.0),
            A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness_limit", 0.2),
                contrast_limit=aug.get("contrast_limit", 0.2),
                p=0.4,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class DeepfakeDataset(Dataset):
    """
    Reads images listed in a CSV manifest.

    Manifest columns:
        path  – file path (absolute, or relative to data_dir)
        label – 0 = real, 1 = fake
    """

    def __init__(self, manifest_path: str, data_dir: str, transform: A.Compose):
        self.data_dir  = data_dir
        self.transform = transform
        self.df = pd.read_csv(manifest_path)

        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("Manifest CSV must contain 'path' and 'label' columns.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label    = int(row["label"])

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.data_dir, img_path)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        tensor    = augmented["image"]

        return tensor, label
