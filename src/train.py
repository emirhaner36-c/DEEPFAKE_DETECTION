"""
Training loop with:
  - Mixed precision (torch.cuda.amp)
  - Cosine LR scheduler with linear warmup
  - Early stopping
  - Best-checkpoint saving
"""
import os
import math
import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Learning-rate schedule ────────────────────────────────────────────────────

def get_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    train_cfg   = cfg["training"]
    total_epochs = train_cfg["epochs"]
    warmup_steps = train_cfg.get("warmup_epochs", 2) * steps_per_epoch
    total_steps  = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    training: bool,
    grad_accum_steps: int = 1,
) -> Tuple[float, float]:
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    amp_device = device.type if device.type == "cuda" else "cpu"

    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(tqdm(loader, leave=False, desc="train" if training else "val")):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(amp_device, enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, labels) / grad_accum_steps

            if training:
                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            total_loss += loss.item() * grad_accum_steps * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Full training loop ────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_dir: str,
    log_dir: str,
) -> dict:
    train_cfg = cfg["training"]
    epochs           = train_cfg["epochs"]
    use_amp          = train_cfg.get("use_amp", True) and device.type == "cuda"
    patience         = train_cfg.get("early_stopping_patience", 5)
    grad_accum_steps = train_cfg.get("grad_accumulation_steps", 1)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scaler    = GradScaler("cuda", enabled=use_amp)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_val_loss = float("inf")
    no_improve    = 0
    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    log_file = open(os.path.join(log_dir, "training_log.csv"), "w")
    log_file.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, use_amp, training=True,
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scheduler,
            scaler, device, use_amp, training=False,
        )

        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}"
        )
        logger.info(
            f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}  lr: {current_lr:.2e}"
        )

        log_file.write(
            f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{current_lr:.8f}\n"
        )
        log_file.flush()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            ckpt_path     = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc":  val_acc,
                    "cfg":      cfg,
                },
                ckpt_path,
            )
            logger.info(f"  Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            logger.info(f"  No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    log_file.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return history
