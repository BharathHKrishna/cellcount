"""Training loop for the U-Net nucleus-segmentation model, logged to W&B."""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from segment.unet_model import UNet  # noqa: E402
from dataset import DSBNucleiDataset  # noqa: E402


def dice_coefficient(pred_probs, target, eps=1e-6):
    pred = (pred_probs > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def iou_score(pred_probs, target, eps=1e-6):
    pred = (pred_probs > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def train(config_path="config.yaml", max_epochs=None, wandb_mode=None, device=None):
    """Train U-Net on DSBNucleiDataset per config_path.

    Logs loss and Dice/IoU (train + val) to Weights & Biases each epoch, plus
    a handful of sample predictions as images. Saves the best checkpoint to
    ../models/unet_v1.pt.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(config_path)
    data_dir = os.path.normpath(os.path.join(base_dir, config["dataset"]["data_dir"]))
    image_size = config["dataset"]["image_size"]

    full_dataset = DSBNucleiDataset(data_dir, image_size=image_size)
    max_images = config["dataset"].get("max_images")
    if max_images is not None and max_images < len(full_dataset):
        rng = random.Random(seed)
        subset_ids = rng.sample(full_dataset.image_ids, max_images)
        full_dataset = DSBNucleiDataset(data_dir, image_size=image_size, image_ids=subset_ids)
    n = len(full_dataset)
    n_val = max(1, int(n * config["dataset"]["val_split"]))
    n_test = max(1, int(n * config["dataset"]["test_split"]))
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

    batch_size = config["train"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_filters=config["model"]["base_filters"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    epochs = max_epochs or config["train"]["epochs"]

    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config=config,
        mode=wandb_mode or "online",
    )

    best_val_dice = -1.0
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "unet_v1.pt")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        sample_images = []
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                probs = torch.sigmoid(logits)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_coefficient(probs, masks) * images.size(0)
                val_iou += iou_score(probs, masks) * images.size(0)

                if batch_idx == 0:
                    for i in range(min(4, images.size(0))):
                        sample_images.append(wandb.Image(
                            images[i].cpu().permute(1, 2, 0).numpy(),
                            masks={
                                "prediction": {"mask_data": (probs[i, 0].cpu().numpy() > 0.5).astype(np.uint8)},
                                "ground_truth": {"mask_data": masks[i, 0].cpu().numpy().astype(np.uint8)},
                            },
                        ))

        val_loss /= len(val_set)
        val_dice /= len(val_set)
        val_iou /= len(val_set)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/dice": val_dice,
            "val/iou": val_iou,
            "val/samples": sample_images,
        })

        print(f"epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "in_channels": config["model"]["in_channels"],
                    "out_channels": config["model"]["out_channels"],
                    "base_filters": config["model"]["base_filters"],
                    "image_size": image_size,
                },
                "val_dice": val_dice,
                "val_iou": val_iou,
                "epoch": epoch,
            }, checkpoint_path)

    run.finish()
    return checkpoint_path, {"best_val_dice": best_val_dice, "test_set": test_set}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    train(config_path=args.config, max_epochs=args.epochs, wandb_mode=args.wandb_mode)
