"""Training loop for the U-Net nucleus-segmentation model, logged to W&B."""


def train(config_path="config.yaml"):
    """Train U-Net on DSBNucleiDataset per config_path.

    Logs loss and Dice/IoU (train + val) to Weights & Biases each epoch, plus
    a handful of sample predictions as images. Saves the best checkpoint to
    ../models/unet_v1.pt.
    """
    raise NotImplementedError("load config, build DSBNucleiDataset + DataLoader, "
                               "UNet model, optimizer, wandb.init, training loop")


if __name__ == "__main__":
    train()
