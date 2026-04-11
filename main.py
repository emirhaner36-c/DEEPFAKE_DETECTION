"""
Deepfake Detection — main entry point.

Modes:
  train    – train the model from scratch
  evaluate – evaluate best checkpoint on test set
  predict  – run inference on a single image or directory
  gradcam  – generate Grad-CAM visualisations

Usage examples:
  python main.py train
  python main.py evaluate
  python main.py predict --input path/to/image.jpg
  python main.py predict --input path/to/folder/
  python main.py gradcam --input path/to/image.jpg
"""
import argparse
import logging
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("Using CPU.")
    return torch.device("cpu")


# ── Train ──────────────────────────────────────────────────────────────────────

def cmd_train(cfg: dict, device: torch.device) -> None:
    from src.model   import build_model
    from src.dataset import DeepfakeDataset, build_transforms
    from src.train   import train
    from src.evaluate import run_evaluation

    image_size = cfg["data"]["image_size"]

    train_dataset = DeepfakeDataset(
        manifest_path=cfg["data"]["manifest_train"],
        data_dir=cfg["data"]["data_dir"],
        transform=build_transforms(image_size, "train", cfg),
    )
    val_dataset = DeepfakeDataset(
        manifest_path=cfg["data"]["manifest_val"],
        data_dir=cfg["data"]["data_dir"],
        transform=build_transforms(image_size, "val", cfg),
    )

    nw      = cfg["training"]["num_workers"]
    bs      = cfg["training"]["batch_size"]
    pin_mem = device.type == "cuda"  # pin_memory not supported on MPS

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin_mem, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin_mem)

    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    model   = build_model(cfg, device)
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        checkpoint_dir=cfg["paths"]["checkpoints"],
        log_dir=cfg["paths"]["logs"],
    )

    # Auto-evaluate on test set after training
    test_dataset = DeepfakeDataset(
        manifest_path=cfg["data"]["manifest_test"],
        data_dir=cfg["data"]["data_dir"],
        transform=build_transforms(image_size, "test", cfg),
    )
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pin_mem)

    # Load best checkpoint for evaluation
    best_ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pt")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    run_evaluation(model, test_loader, device, cfg["paths"]["results"], history)


# ── Evaluate ───────────────────────────────────────────────────────────────────

def cmd_evaluate(cfg: dict, device: torch.device) -> None:
    from src.model    import build_model
    from src.dataset  import DeepfakeDataset, build_transforms
    from src.evaluate import run_evaluation
    from src.predict  import load_checkpoint

    image_size = cfg["data"]["image_size"]
    best_ckpt  = os.path.join(cfg["paths"]["checkpoints"], "best_model.pt")

    model = load_checkpoint(best_ckpt, cfg, device)

    test_dataset = DeepfakeDataset(
        manifest_path=cfg["data"]["manifest_test"],
        data_dir=cfg["data"]["data_dir"],
        transform=build_transforms(image_size, "test", cfg),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    run_evaluation(model, test_loader, device, cfg["paths"]["results"])


# ── Predict ────────────────────────────────────────────────────────────────────

def cmd_predict(cfg: dict, device: torch.device, input_path: str, threshold: float) -> None:
    from src.predict import load_checkpoint, predict_image, predict_batch

    best_ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pt")
    model     = load_checkpoint(best_ckpt, cfg, device)
    img_size  = cfg["data"]["image_size"]

    if os.path.isdir(input_path):
        exts  = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in exts
        ]
        results = predict_batch(model, paths, device, img_size, threshold)
    else:
        results = [predict_image(model, input_path, device, img_size, threshold)]

    for r in results:
        print(f"{os.path.basename(r['image']):<40}  {r['label']:<6}  "
              f"real={r['prob_real']:.2%}  fake={r['prob_fake']:.2%}")


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

def cmd_gradcam(cfg: dict, device: torch.device, input_path: str) -> None:
    from src.predict        import load_checkpoint
    from src.explainability import generate_gradcam, batch_gradcam

    best_ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pt")
    model     = load_checkpoint(best_ckpt, cfg, device)
    img_size  = cfg["data"]["image_size"]
    out_dir   = os.path.join(cfg["paths"]["results"], "gradcam")

    if os.path.isdir(input_path):
        exts  = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in exts
        ]
        batch_gradcam(model, paths, out_dir, img_size, device)
    else:
        fname     = os.path.splitext(os.path.basename(input_path))[0]
        save_path = os.path.join(out_dir, f"{fname}_gradcam.png")
        result    = generate_gradcam(model, input_path, save_path, img_size, device)
        print(f"Prediction: {result['predicted_label']}  ({result['confidence']:.1%} confidence)")
        print(f"Grad-CAM saved → {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection System")
    parser.add_argument("mode", choices=["train", "evaluate", "predict", "gradcam"])
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--input",     default=None, help="Image path or folder (predict/gradcam)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device(force_cpu=True)  # MPS has memory issues with EfficientNet-B4; use Colab for GPU

    if args.mode == "train":
        cmd_train(cfg, device)

    elif args.mode == "evaluate":
        cmd_evaluate(cfg, device)

    elif args.mode == "predict":
        if not args.input:
            parser.error("--input required for predict mode")
        cmd_predict(cfg, device, args.input, args.threshold)

    elif args.mode == "gradcam":
        if not args.input:
            parser.error("--input required for gradcam mode")
        cmd_gradcam(cfg, device, args.input)


if __name__ == "__main__":
    main()
