"""2018 Data Science Bowl / BBBC038 nuclei dataset loader."""


class DSBNucleiDataset:
    """PyTorch Dataset over BBBC038 images + per-image instance masks, merged
    to a single binary foreground mask for U-Net training.

    data_dir: path to the extracted DSB2018 stage1_train layout
        (<data_dir>/<image_id>/images/<image_id>.png,
         <data_dir>/<image_id>/masks/*.png).
    """

    def __init__(self, data_dir, image_size=256, transform=None):
        raise NotImplementedError("index image_ids under data_dir, merge per-instance "
                                   "masks into one binary mask per image")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError("return (image_tensor, binary_mask_tensor)")
