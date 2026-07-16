"""v2: your own trained U-Net, as an alternative to the Cellpose baseline."""

import numpy as np
import torch
import torch.nn as nn

from .postprocess import separate_instances


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Standard 4-level U-Net for binary foreground/background segmentation."""

    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


_RUN_CACHE = {}


def _prepare_image(image, image_size):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    image -= image.min()
    max_val = image.max()
    if max_val > 0:
        image /= max_val

    import cv2
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    return tensor, image.shape[:2]


def run_unet(image, weights_path="models/unet_v1.pt", threshold=0.5):
    """Run the trained U-Net on an image and return an instance-labeled mask.

    image: numpy array (H, W) or (H, W, C).
    Returns: numpy array (H, W) of integer instance labels (0 = background),
    after watershed/connected-components instance separation on the raw mask.
    """
    if weights_path not in _RUN_CACHE:
        checkpoint = torch.load(weights_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model = UNet(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 1),
            base_filters=config.get("base_filters", 32),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _RUN_CACHE[weights_path] = (model, config.get("image_size", 256))
    model, image_size = _RUN_CACHE[weights_path]

    tensor, orig_shape = _prepare_image(image, image_size)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].numpy()

    import cv2
    prob_full = cv2.resize(prob, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
    binary_mask = prob_full > threshold
    return separate_instances(binary_mask)
