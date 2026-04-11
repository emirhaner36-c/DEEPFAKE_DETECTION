"""
Single-image and batch inference.
"""
import os
import logging
from typing import Union

import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import DeepfakeClassifier, build_model

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Real", 1: "Fake"}


def load_checkpoint(checkpoint_path: str, cfg: dict, device: torch.device) -> DeepfakeClassifier:
    model = build_model(cfg, device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}  (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return model


def preprocess_image(image_path: str, image_size: int) -> torch.Tensor:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return transform(image=img)["image"].unsqueeze(0)


def predict_image(
    model: DeepfakeClassifier,
    image_path: str,
    device: torch.device,
    image_size: int = 380,
    threshold: float = 0.5,
) -> dict:
    """
    Predict whether a single image is real or deepfake.

    Returns:
        {
            "image":       path,
            "label":       "Real" | "Fake",
            "class_id":    0 | 1,
            "prob_real":   float,
            "prob_fake":   float,
        }
    """
    tensor = preprocess_image(image_path, image_size).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    prob_real = probs[0].item()
    prob_fake = probs[1].item()
    class_id  = 1 if prob_fake >= threshold else 0

    return {
        "image":     image_path,
        "label":     CLASS_NAMES[class_id],
        "class_id":  class_id,
        "prob_real": round(prob_real, 4),
        "prob_fake": round(prob_fake, 4),
    }


def predict_batch(
    model: DeepfakeClassifier,
    image_paths: list[str],
    device: torch.device,
    image_size: int = 380,
    threshold: float = 0.5,
) -> list[dict]:
    """Run predict_image over a list of paths."""
    results = []
    for path in image_paths:
        result = predict_image(model, path, device, image_size, threshold)
        logger.info(
            f"{os.path.basename(path)}: {result['label']}  "
            f"(real={result['prob_real']:.2%}, fake={result['prob_fake']:.2%})"
        )
        results.append(result)
    return results
