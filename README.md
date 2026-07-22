---
title: CellCount
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# CellCount

Upload a microscopy image and the model outlines and counts every individual cell.

See [CLAUDE.md](./CLAUDE.md) for the full project brief, model trade-offs, dataset links, and build order.

## Status

✅ Weekend 1 — Cellpose baseline + Gradio demo.
✅ Weekend 2 — U-Net trained on BBBC038/DSB2018 nuclei, logged to W&B (val Dice 0.867).
✅ Weekend 3 — Pretrained-vs-My-U-Net comparison, per-cell measurements, CSV export.
✅ Weekend 4 — README, metrics, limitations, live deployment.

## Live demo

**[cellcount.onrender.com](https://cellcount.onrender.com)** — runs a lite build (U-Net only; Cellpose's SAM backbone needs more RAM than Render's free tier gives). Free tier sleeps after 15 min idle, so the first load can take up to a minute to wake up.

For the full Cellpose-vs-U-Net comparison, run locally: `python app.py` (Gradio) or `uvicorn app.api.main:app` (FastAPI).

## How it works

1. Upload a microscopy image (cells or nuclei).
2. The app segments it with either:
   - **Cellpose (pretrained)** — a generalist, state-of-the-art segmentation model (`cellpose-sam`), used out of the box.
   - **My U-Net** — a U-Net (PyTorch) trained from scratch on the [2018 Data Science Bowl / BBBC038](https://bbbc.broadinstitute.org/BBBC038) nuclei dataset.
3. Touching cells are separated into instances (watershed on the distance transform for the U-Net's binary mask; Cellpose returns instances natively).
4. Outlines are drawn over the original image, per-cell measurements (area, centroid) are computed, and a downloadable CSV + size-distribution histogram are produced.

## Metrics

U-Net trained for 8 epochs on a 210/45/45 train/val/test split of a 300-image subset of BBBC038 stage-1 training images (image-level binary masks, 128x128), logged to Weights & Biases. Best checkpoint: val Dice 0.867 / val IoU 0.775 (epoch 7).

| Model | Dice | IoU | Count MAE |
|---|---|---|---|
| My U-Net | 0.789 | 0.694 | 2.47 |
| Cellpose (pretrained, generalist) | 0.783 | 0.687 | 2.47 |

Evaluated head-to-head on 15 held-out test images with [train/eval.py](./train/eval.py) (sample size kept small — Cellpose's SAM backbone is slow on CPU). The U-Net edges out the Cellpose baseline here — a real result, but not one to over-read: it's 8 epochs on 300 images vs. a generalist foundation model, evaluated on a small sample, and an earlier version of the watershed instance-separation step had a real over-segmentation bug (fixed by adding minimum-size filtering and a per-blob fallback seed so small real nuclei aren't dropped — see [app/segment/postprocess.py](./app/segment/postprocess.py)) that inflated count error before this fix. Treat this as "competitive with a strong baseline on this dataset," not "beats Cellpose in general." W&B run: see `train/wandb/` (offline runs; sync with `wandb sync`).

## Limitations

- The U-Net was trained on a single held-out split of BBBC038 nuclei images and has not been validated on other tissue/cell types — it is not a generalist model the way Cellpose is.
- Training used a deliberately modest budget (300/670 available images, 128x128 input, 8 epochs, ~16 base filters) to keep training time reasonable on CPU. Accuracy would likely improve with the full dataset, longer training, augmentation, and a larger input resolution.
- Densely overlapping or touching cells are the main failure mode for both models; the U-Net's watershed-based instance separation is still more sensitive to under/over-segmentation on crowded fields than Cellpose's learned flow fields, even after the fix above.
- Low-contrast or unusually stained images (outside the DSB2018 distribution) degrade U-Net accuracy more than Cellpose, which was trained on a broader, more varied dataset.
- The head-to-head eval used only 15 test images (not the full 45-image test split) since Cellpose's SAM backbone is slow on CPU — the comparison numbers should be read as indicative, not a tightly-powered benchmark.

## Datasets

- **2018 Data Science Bowl / BBBC038** ([Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC038), CC0) — nucleus segmentation, used to train the U-Net.

## Repository structure

See [CLAUDE.md](./CLAUDE.md#repository-structure).

## Running locally

```bash
pip install -r requirements.txt
python app.py                 # Gradio app on localhost
# or
pip install -r app/requirements.txt
uvicorn app.api.main:app --reload --app-dir app   # FastAPI version
```

To retrain the U-Net:

```bash
cd train
python train_unet.py --wandb-mode online   # or offline / disabled
python eval.py                              # Dice/IoU/count MAE, U-Net vs Cellpose
```
