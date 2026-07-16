# models/

Trained weights live here, gitignored (see `.gitignore` — track via DVC or a release asset instead of git).

- `unet_v1.pt` — U-Net trained on a 300-image subset of BBBC038/DSB2018 nuclei (`train/train_unet.py`, 8 epochs, 128x128, base_filters=16). Best checkpoint: val Dice 0.867 / val IoU 0.775 (epoch 7). See [README.md](../README.md#metrics) for the head-to-head vs Cellpose.
