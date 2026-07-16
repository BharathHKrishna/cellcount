"""Evaluate a trained model: Dice/IoU and count accuracy vs ground truth,
plus a head-to-head against the Cellpose baseline."""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from segment.postprocess import measure_instances  # noqa: E402
from segment.cellpose_runner import run_cellpose  # noqa: E402
from segment.unet_model import run_unet  # noqa: E402
from dataset import DSBNucleiDataset  # noqa: E402


def _dice_iou(pred_binary, true_binary, eps=1e-6):
    pred_binary = pred_binary.astype(bool)
    true_binary = true_binary.astype(bool)
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    dice = (2 * intersection + eps) / (pred_binary.sum() + true_binary.sum() + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def evaluate(model_predict_fn, dataset, name="model"):
    """Run model_predict_fn over dataset and report Dice, IoU, and count MAE.

    model_predict_fn: callable(image) -> binary/instance mask, e.g. run_unet or
        a wrapper around run_cellpose.
    dataset: held-out DSBNucleiDataset (or similar) with ground-truth masks.
    Returns: dict with mean_dice, mean_iou, count_mae.
    """
    dices, ious, count_errors = [], [], []

    for idx in range(len(dataset)):
        image_tensor, mask_tensor = dataset[idx]
        image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        true_binary = mask_tensor[0].numpy() > 0.5
        true_count = len(measure_instances(_label_binary(true_binary)))

        pred_instance_mask = model_predict_fn(image)
        pred_binary = np.asarray(pred_instance_mask) > 0
        pred_count = len(measure_instances(pred_instance_mask))

        dice, iou = _dice_iou(pred_binary, true_binary)
        dices.append(dice)
        ious.append(iou)
        count_errors.append(abs(pred_count - true_count))

    result = {
        "name": name,
        "mean_dice": float(np.mean(dices)),
        "mean_iou": float(np.mean(ious)),
        "count_mae": float(np.mean(count_errors)),
        "n_images": len(dataset),
    }
    print(f"{name}: dice={result['mean_dice']:.4f} iou={result['mean_iou']:.4f} "
          f"count_mae={result['count_mae']:.2f} (n={result['n_images']})")
    return result


def _label_binary(binary_mask):
    from scipy import ndimage as ndi
    labeled, _ = ndi.label(binary_mask)
    return labeled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default="../models/unet_v1.pt")
    parser.add_argument("--n", type=int, default=None, help="limit number of test images")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(args.config)
    data_dir = os.path.normpath(os.path.join(base_dir, config["dataset"]["data_dir"]))
    image_size = config["dataset"]["image_size"]

    full_dataset = DSBNucleiDataset(data_dir, image_size=image_size)
    n = len(full_dataset)
    n_val = max(1, int(n * config["dataset"]["val_split"]))
    n_test = max(1, int(n * config["dataset"]["test_split"]))
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(config["train"]["seed"])
    _, _, test_set = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

    if args.n is not None:
        test_set = torch.utils.data.Subset(test_set, range(min(args.n, len(test_set))))

    weights_path = os.path.normpath(os.path.join(base_dir, args.weights))
    unet_result = evaluate(lambda img: run_unet(img, weights_path=weights_path), test_set, name="unet")
    cellpose_result = evaluate(lambda img: run_cellpose(img), test_set, name="cellpose")

    print("\n=== Comparison ===")
    print(f"{'metric':<12}{'unet':>10}{'cellpose':>12}")
    for key in ("mean_dice", "mean_iou", "count_mae"):
        print(f"{key:<12}{unet_result[key]:>10.4f}{cellpose_result[key]:>12.4f}")
