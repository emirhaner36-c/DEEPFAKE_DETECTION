"""
Deepfake Detection Demo
-----------------------
Gradio app that classifies an uploaded face image as Real or Fake
and shows a Grad-CAM heatmap highlighting the regions the model focused on.

Usage:
    python demo/app.py
    python demo/app.py --checkpoint path/to/best_model.pt
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow running from project root or demo/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DeepfakeClassifier

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 380          # must match the Colab training resolution
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLASS_NAMES  = {0: "Real", 1: "Fake"}
CLASS_COLORS = {0: "#22c55e", 1: "#ef4444"}   # green / red

DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "demo", "best_model.pt",
)

# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> DeepfakeClassifier:
    model = DeepfakeClassifier(
        backbone="tf_efficientnet_b4.ns_jft_in1k",
        num_classes=2,
        pretrained=False,
        dropout=0.3,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


# ── Inference + Grad-CAM ───────────────────────────────────────────────────────
def _preprocess(img_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """Return (float32 image in [0,1], normalised tensor)."""
    resized   = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_float = resized.astype(np.float32) / 255.0
    transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    tensor = transform(image=resized)["image"].unsqueeze(0)
    return img_float, tensor


def predict_and_explain(
    pil_image: Image.Image,
    model: DeepfakeClassifier,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    """
    Returns:
        label_dict  – Gradio Label component dict  {label: confidence, ...}
        gradcam_img – RGB numpy array for Gradio Image component
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img_float, tensor = _preprocess(img_rgb)
    tensor = tensor.to(device)

    # Prediction
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_class = probs.argmax().item()
    prob_fake  = probs[1].item()
    prob_real  = probs[0].item()

    # Grad-CAM
    with GradCAM(model=model, target_layers=[model.get_feature_layer()]) as cam:
        mask = cam(tensor, targets=[ClassifierOutputTarget(pred_class)])[0]

    vis = show_cam_on_image(img_float, mask, use_rgb=True)

    label_dict = {
        "Real": round(prob_real, 4),
        "Fake": round(prob_fake, 4),
    }
    return label_dict, vis


# ── Gradio UI ──────────────────────────────────────────────────────────────────
def build_ui(model: DeepfakeClassifier, device: torch.device) -> gr.Blocks:
    def run(pil_image):
        if pil_image is None:
            return None, None
        label_dict, gradcam_img = predict_and_explain(pil_image, model, device)
        pred  = "REAL" if label_dict["Real"] > label_dict["Fake"] else "FAKE"
        conf  = max(label_dict["Real"], label_dict["Fake"])
        color = "#22c55e" if pred == "REAL" else "#ef4444"
        result_html = f'<div style="text-align:center;font-size:2.5rem;font-weight:700;color:{color}">{pred}</div>' \
                      f'<div style="text-align:center;font-size:1.1rem;color:#888">{conf:.1%} confidence</div>'
        return result_html, gradcam_img

    with gr.Blocks(title="Deepfake Detector") as demo:
        gr.Markdown("## Deepfake Detector")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload face image")
            with gr.Column():
                result_out  = gr.HTML()
                gradcam_out = gr.Image(type="numpy", label="Grad-CAM")

        image_input.change(fn=run, inputs=image_input, outputs=[result_out, gradcam_out])

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CKPT,
        help="Path to best_model.pt (default: checkpoints/demo/best_model.pt)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Pass --checkpoint path/to/best_model.pt")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model on {device} from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)
    print("Model ready.")

    app = build_ui(model, device)
    app.launch(server_port=args.port, share=args.share,
               theme=gr.themes.Soft(), css="footer{display:none}")


if __name__ == "__main__":
    main()
