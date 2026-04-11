"""
Grad-CAM visualisation for the deepfake classifier.

Highlights which facial regions most influenced the model's prediction.
Requires: pip install grad-cam
"""
import os
import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Real", 1: "Fake"}


def _load_image(image_path: str, image_size: int) -> tuple[np.ndarray, torch.Tensor]:
    """Load image and return both the normalised float array and the model input tensor."""
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (image_size, image_size))

    # Float array in [0,1] — used for overlay
    img_float = img_rgb.astype(np.float32) / 255.0

    transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    tensor = transform(image=img_rgb)["image"].unsqueeze(0)  # (1, C, H, W)
    return img_float, tensor


def generate_gradcam(
    model: torch.nn.Module,
    image_path: str,
    save_path: str,
    image_size: int = 380,
    device: torch.device = torch.device("cpu"),
    target_class: Optional[int] = None,
) -> dict:
    """
    Generate a Grad-CAM heatmap for a single image.

    Args:
        model        – trained DeepfakeClassifier
        image_path   – path to the input image
        save_path    – where to save the visualisation
        image_size   – must match training resolution
        device       – cpu / cuda
        target_class – 0 (real) or 1 (fake). If None, uses predicted class.

    Returns:
        dict with keys: predicted_class, predicted_label, confidence
    """
    model.eval()
    img_float, tensor = _load_image(image_path, image_size)
    tensor = tensor.to(device)

    # ── Get prediction ──────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    pred_class  = probs.argmax().item()
    confidence  = probs[pred_class].item()

    if target_class is None:
        target_class = pred_class

    # ── Grad-CAM ────────────────────────────────────────────────────────────
    target_layer = [model.get_feature_layer()]
    targets      = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # (H, W)

    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # ── Save side-by-side plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow((img_float * 255).astype(np.uint8))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(visualization)
    axes[1].set_title(
        f"Grad-CAM → {CLASS_NAMES[pred_class]} ({confidence:.1%})"
    )
    axes[1].axis("off")

    plt.suptitle(
        f"Prediction: {CLASS_NAMES[pred_class]}  |  Confidence: {confidence:.1%}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Grad-CAM saved → {save_path}")

    return {
        "predicted_class": pred_class,
        "predicted_label": CLASS_NAMES[pred_class],
        "confidence":      round(confidence, 4),
    }


def batch_gradcam(
    model: torch.nn.Module,
    image_paths: list[str],
    output_dir: str,
    image_size: int = 380,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Run Grad-CAM on a list of images and save visualisations."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, path in enumerate(image_paths):
        fname     = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(output_dir, f"{fname}_gradcam.png")
        result    = generate_gradcam(model, path, save_path, image_size, device)
        result["image"] = path
        results.append(result)
        logger.info(f"[{i+1}/{len(image_paths)}] {fname}: {result['predicted_label']} ({result['confidence']:.1%})")

    return results
