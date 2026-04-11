"""
End-to-end smoke test using synthetic (random) images.
Verifies the full pipeline: dataset → model → train → evaluate → grad-cam
without needing real data.
"""
import os, sys, shutil, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(__file__))

# ── Generate synthetic face-like images ───────────────────────────────────────
def make_synthetic_dataset(n=80, img_size=380):
    img_dir = os.path.join(ROOT, "data", "processed", "smoke")
    os.makedirs(os.path.join(img_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "fake"), exist_ok=True)

    records = []
    for i in range(n):
        label = i % 2           # alternate real / fake
        folder = "real" if label == 0 else "fake"
        path   = os.path.join(img_dir, folder, f"img_{i:04d}.jpg")
        # random colour image
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        cv2.imwrite(path, img)
        records.append({"path": path, "label": label})

    random.shuffle(records)
    df = pd.DataFrame(records)

    manifest_dir = os.path.join(ROOT, "data", "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)

    df.iloc[:n_train       ].to_csv(os.path.join(manifest_dir, "train.csv"), index=False)
    df.iloc[n_train:n_train+n_val].to_csv(os.path.join(manifest_dir, "val.csv"),   index=False)
    df.iloc[n_train+n_val: ].to_csv(os.path.join(manifest_dir, "test.csv"),  index=False)

    print(f"Synthetic dataset: {n} images ({n_train} train / {n_val} val / {n-n_train-n_val} test)")
    return records[0]["path"]   # return one path for grad-cam test


def main():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    # Tiny settings for fast smoke test
    cfg["training"]["epochs"]        = 2
    cfg["training"]["batch_size"]    = 8
    cfg["training"]["num_workers"]   = 0
    cfg["training"]["use_amp"]       = False
    cfg["data"]["data_dir"]          = "data/processed/smoke"
    cfg["paths"]["checkpoints"]      = "checkpoints/smoke"
    cfg["paths"]["results"]          = "results/smoke"
    cfg["paths"]["logs"]             = "logs/smoke"

    device = torch.device("cpu")

    print("\n[1/5] Generating synthetic images...")
    sample_img = make_synthetic_dataset(n=80, img_size=380)

    print("\n[2/5] Building model (EfficientNet-B4, pretrained=False for speed)...")
    cfg["model"]["pretrained"] = False
    from src.model import build_model
    model = build_model(cfg, device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"       Model parameters: {params:.1f}M")

    print("\n[3/5] Running training loop (2 epochs)...")
    from torch.utils.data import DataLoader
    from src.dataset import DeepfakeDataset, build_transforms
    from src.train   import train

    image_size   = cfg["data"]["image_size"]
    train_loader = DataLoader(
        DeepfakeDataset("data/manifests/train.csv", cfg["data"]["data_dir"],
                        build_transforms(image_size, "train", cfg)),
        batch_size=8, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        DeepfakeDataset("data/manifests/val.csv", cfg["data"]["data_dir"],
                        build_transforms(image_size, "val", cfg)),
        batch_size=8, shuffle=False, num_workers=0,
    )

    history = train(model, train_loader, val_loader, cfg, device,
                    cfg["paths"]["checkpoints"], cfg["paths"]["logs"])

    print("\n[4/5] Running evaluation...")
    from src.predict  import load_checkpoint
    from src.evaluate import run_evaluation

    model = load_checkpoint(os.path.join(cfg["paths"]["checkpoints"], "best_model.pt"), cfg, device)
    test_loader = DataLoader(
        DeepfakeDataset("data/manifests/test.csv", cfg["data"]["data_dir"],
                        build_transforms(image_size, "test", cfg)),
        batch_size=8, shuffle=False, num_workers=0,
    )
    metrics = run_evaluation(model, test_loader, device, cfg["paths"]["results"], history)

    print("\n[5/5] Running Grad-CAM...")
    from src.explainability import generate_gradcam
    out_path = os.path.join(cfg["paths"]["results"], "gradcam", "smoke_test.png")
    result   = generate_gradcam(model, sample_img, out_path, image_size, device)
    print(f"       Prediction: {result['predicted_label']} ({result['confidence']:.1%})")

    print("\n" + "="*50)
    print("  SMOKE TEST PASSED — full pipeline working")
    print("="*50)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  Plots    : {cfg['paths']['results']}/plots/")
    print(f"  Grad-CAM : {out_path}")


if __name__ == "__main__":
    main()
