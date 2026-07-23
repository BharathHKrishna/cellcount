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
✅ Weekend 2 — U-Net trained on BBBC038/DSB2018 nuclei, logged to W&B (val Dice 0.905).
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

U-Net trained for 18 epochs on a 469/100/101 train/val/test split of the full 670-image BBBC038 stage-1 training set (image-level binary masks, 128x128), logged to Weights & Biases. Best checkpoint: val Dice 0.905 / val IoU 0.838 (epoch 16).

| Model | Dice | IoU | Count MAE |
|---|---|---|---|
| My U-Net | 0.808 | 0.716 | 2.93 |
| Cellpose (pretrained, generalist) | 0.783 | 0.687 | 2.47 |

Evaluated head-to-head on 15 held-out test images with [train/eval.py](./train/eval.py) (sample size kept small — Cellpose's SAM backbone is slow on CPU). This is a real trade-off, not a clean win: the U-Net now beats the Cellpose baseline on Dice and IoU (better pixel-level mask quality — it covers more of the true foreground correctly), but is slightly behind on count MAE (it occasionally over- or under-splits touching nuclei by one or two, where Cellpose's learned flow fields are more consistent). An earlier, smaller training run (300 images, 8 epochs) was closer to Cellpose on count but weaker on Dice/IoU; this version was chosen because a nucleus going undetected entirely (a Dice/IoU problem) is a worse failure mode for this tool than an off-by-one instance count. Don't over-read either way: it's a from-scratch model trained in minutes on a laptop CPU vs. a generalist foundation model, evaluated on a small sample. W&B run: see `train/wandb/` (offline runs; sync with `wandb sync`).

## Limitations

- **The U-Net misses some nuclei outright on certain images** — confirmed by a user against the live demo (a small crop with a mix of a textured/mottled clump and a smooth bright nucleus; the latter type went undetected). Retraining on the full dataset (see Metrics) improved held-out Dice/IoU substantially and was a good-faith attempt at this exact problem, but it was never confirmed against the original image — Gradio doesn't persist uploads server-side, so the report couldn't be directly reproduced or verified fixed. Treat "handles all nucleus appearances" as unproven; if you hit this, the most useful thing you can do is send the actual image file (not a screenshot) so it can be reproduced.
- The U-Net was trained on BBBC038 nuclei only and has not been validated on other tissue/cell types — it is not a generalist model the way Cellpose is.
- Training used a deliberately modest budget (128x128 input, ~16 base filters, no augmentation) to keep training time reasonable on CPU. Accuracy would likely improve further with augmentation, a larger input resolution, and more capacity.
- Densely overlapping or touching cells are a failure mode for both models; the U-Net's watershed-based instance separation is more sensitive to under/over-segmentation on crowded fields than Cellpose's learned flow fields, even after fixing an earlier over-segmentation bug (see git history on `app/segment/postprocess.py`).
- Low-contrast or unusually stained images (outside the DSB2018 distribution) degrade U-Net accuracy more than Cellpose, which was trained on a broader, more varied dataset — this is the likely underlying cause of the missed-nuclei issue above.
- The head-to-head eval used only 15 test images (not the full 101-image test split) since Cellpose's SAM backbone is slow on CPU — the comparison numbers should be read as indicative, not a tightly-powered benchmark.
- The U-Net beats Cellpose on Dice/IoU but loses on count MAE (see Metrics) — it's better at covering the right pixels than at getting the exact instance count right on every image.

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
