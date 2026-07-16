"""Evaluate a trained model: Dice/IoU and count accuracy vs ground truth,
plus a head-to-head against the Cellpose baseline."""


def evaluate(model_predict_fn, dataset, name="model"):
    """Run model_predict_fn over dataset and report Dice, IoU, and count MAE.

    model_predict_fn: callable(image) -> binary/instance mask, e.g. run_unet or
        a wrapper around run_cellpose.
    dataset: held-out DSBNucleiDataset (or similar) with ground-truth masks.
    Returns: dict with mean_dice, mean_iou, count_mae.
    """
    raise NotImplementedError("per-image Dice/IoU against ground-truth binary mask, "
                               "instance count vs true instance count -> MAE")


if __name__ == "__main__":
    raise NotImplementedError("load test split, evaluate U-Net and Cellpose, print/compare")
