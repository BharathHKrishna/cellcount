"""2018 Data Science Bowl / BBBC038 nuclei dataset loader."""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DSBNucleiDataset(Dataset):
    """PyTorch Dataset over BBBC038 images + per-image instance masks, merged
    to a single binary foreground mask for U-Net training.

    data_dir: path to the extracted DSB2018 stage1_train layout
        (<data_dir>/<image_id>/images/<image_id>.png,
         <data_dir>/<image_id>/masks/*.png).
    """

    def __init__(self, data_dir, image_size=256, transform=None, image_ids=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.image_ids = image_ids if image_ids is not None else sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_dir = os.path.join(self.data_dir, image_id)

        image_path = os.path.join(image_dir, "images", f"{image_id}.png")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_dir = os.path.join(image_dir, "masks")
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask_file in os.listdir(mask_dir):
            instance = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            binary_mask |= (instance > 0).astype(np.uint8)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        binary_mask = cv2.resize(binary_mask, (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        if self.transform is not None:
            image, binary_mask = self.transform(image, binary_mask)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
        return image_tensor, mask_tensor
