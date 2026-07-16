# CellCount — a cell/nucleus segmentation web app

**One-line pitch:** Upload a microscopy image and the model outlines and counts every individual cell.

---

## What it does (in one breath)

A user uploads a microscopy image (cells or nuclei). The app runs a segmentation model, draws a clean outline around each individual cell, overlays them on the image, and returns a total count plus per-cell measurements (area, etc.). Download the labeled image or a CSV of the counts.

This is your HiWi work — segmentation of cells — turned into a public, deployed tool.

---

## The one decision that matters: which model

You have three sensible paths. Here's the honest trade-off:

| Approach | What it is | Effort | Recommendation |
|---|---|---|---|
| Cellpose (pretrained) | A ready, state-of-the-art generalist cell-segmentation model. You call it, it works on most cell types out of the box. | Low | Start here. Ships in a weekend, looks great immediately. |
| StarDist | Segmentation specialized for star-convex shapes (nuclei). Excellent for nucleus datasets. | Low-medium | Good alternative/addition if you focus on nuclei. |
| Train your own U-Net | Fine-tune a U-Net (or U-Net++) on a public dataset yourself. | Medium-high | Do this in v2 — it's the part that shows you can train models, which is your whole point. |

The smart sequence: ship v1 with Cellpose pretrained (so you have a working, impressive demo fast), then in v2 train your own U-Net on a public dataset and let the user toggle between "pretrained" and "my model" — comparing them. That v2 step is what turns this from "I called an API" into "I trained a segmentation model and benchmarked it against the state of the art." That's the modeling story you want to tell.

---

## Datasets (public, well-known, clean)

- **2018 Data Science Bowl (nuclei segmentation)** — the classic nucleus-segmentation dataset (Kaggle). Thousands of labeled nuclei images across varied conditions. This is your primary training set for the U-Net.
- **Cellpose's own dataset** — generalist cell images with masks; good for evaluation.
- **BBBC (Broad Bioimage Benchmark Collection)** — free, labeled microscopy datasets (e.g. BBBC038 = the Data Science Bowl nuclei). Well-documented, citable.
- **LIVECell** — large-scale label-free cell segmentation dataset if you want something meatier for v2.

Start with the 2018 Data Science Bowl / BBBC038 nuclei set — it's the standard, it's clean, and it's what reviewers will recognize.

---

## Tech stack — every choice with a reason

| Component | Choice | Why |
|---|---|---|
| Segmentation (v1) | Cellpose (pip install) | State-of-the-art, pretrained, works immediately. Instant impressive demo. |
| Segmentation (v2) | U-Net in PyTorch, trained on DSB nuclei | This is the "I train models" showcase — your whole point. |
| Post-processing | scikit-image (watershed, connected components) for instance separation + measurements | Turns a mask into counted, measured individual cells. |
| Backend | FastAPI | Consistent with your other projects; recruiters know it. |
| Frontend | React + a simple upload/canvas overlay (or plain HTML+JS if you want it faster) | Upload → show overlay → show count. Keep it simple. |
| Image handling | Pillow, numpy, OpenCV | Standard. |
| Experiment tracking (v2) | Weights & Biases | You already use it; log the U-Net training. Links from README. |
| Deployment | Hugging Face Spaces (Gradio) OR Vercel frontend + a small GPU/CPU backend | HF Spaces is ideal here — free, ML-native, and it doubles as a public artifact recruiters browse. |
| Data versioning (v2) | DVC | Optional but shows discipline. |

**Deployment recommendation:** Hugging Face Spaces with Gradio for v1. It's the fastest path to a live, shareable ML demo, it's free, and having a Space on your Hugging Face profile is itself a visibility win. You can build the whole v1 as a Gradio app in a weekend. Move to a full FastAPI+React version later if you want it to look more like a product.

---

## Repository structure

```
cellcount/
├── README.md
├── CLAUDE.md
├── app/
│   ├── gradio_app.py          # v1: upload → segment → overlay + count
│   ├── api/                   # v2: FastAPI version
│   │   └── main.py
│   └── segment/
│       ├── cellpose_runner.py # v1 pretrained
│       ├── unet_model.py      # v2 your model
│       ├── postprocess.py     # instance separation, counting, measurements
│       └── overlay.py         # draw outlines + labels
├── train/
│   ├── dataset.py             # DSB / BBBC038 loader
│   ├── train_unet.py          # your training loop (W&B logged)
│   ├── eval.py                # IoU / Dice / count accuracy
│   └── config.yaml
├── models/
│   ├── unet_v1.pt
│   └── README.md
├── notebooks/
│   ├── 01_explore_nuclei_data.ipynb
│   ├── 02_cellpose_baseline.ipynb
│   └── 03_train_unet.ipynb
└── data/                      # DVC-tracked / gitignored
```

---

## Build order — 3 to 4 weekends

**Weekend 1 — Working demo with Cellpose.**
Install Cellpose, feed it a microscopy image, get masks. Write the post-processing (separate touching cells, count them, draw outlines). Wrap it in a Gradio app: upload → see outlined cells + count. Deploy to Hugging Face Spaces. Goal: a live link where anyone can upload a cell image and get a counted, outlined result. This alone is demo-able and screenshot-able.

**Weekend 2 — Train your own U-Net.**
Load the 2018 Data Science Bowl / BBBC038 nuclei dataset. Train a U-Net in PyTorch for nucleus segmentation. Log to W&B (loss, Dice/IoU, sample predictions). Goal: your own trained model with real metrics — the "I build models" proof.

**Weekend 3 — Integrate + compare + measure.**
Add a toggle in the app: "Pretrained (Cellpose)" vs "My U-Net." Show both results side by side with counts. Add per-cell measurements (area, count, maybe size distribution histogram). Goal: it's now a real tool AND a benchmark story.

**Weekend 4 — Polish + README + write-up.**
Clean README with a GIF, your metrics (Dice/IoU, count accuracy vs ground truth), dataset citations, and an honest "limitations" section. Record a 30-second demo. Optional: a short blog/LinkedIn post. Goal: looks like a serious, documented project, not a notebook dump.

---

## The killer demo flow (your 30-second recording)

1. Open the live link. Upload a dense field of nuclei.
2. Model outlines every nucleus, overlays them, shows "247 cells detected."
3. Toggle "My U-Net" vs "Cellpose" — show both, show they agree.
4. Show the size-distribution histogram + downloadable CSV.
5. Upload a different cell type — it still works.

That "247 cells outlined in one click" moment is the visceral, layman-obvious wow.

---

## The honest metrics story (what makes it credible)

Report these in the README — real numbers, honestly:

- **Dice / IoU** of your U-Net on the held-out test set.
- **Count accuracy** — predicted vs true cell count (mean absolute error).
- **Comparison** — your U-Net vs Cellpose baseline (it's fine if Cellpose wins; the point is you built, trained, and honestly benchmarked a model. Reviewers respect characterization over inflated claims).

Add a short "Limitations" section: where it fails (overlapping cells, unusual stains, low contrast). This honesty is exactly what signals research maturity to labs like EMBL/DKFZ.

---

## The resume bullet (once built)

> **CellCount — cell & nucleus segmentation** (Live: [HF Space link])
> Built and deployed a web app that segments and counts individual cells in microscopy images. Trained a U-Net (PyTorch) on the Data Science Bowl nuclei dataset, benchmarked against a Cellpose baseline (Dice [X], count MAE [Y]), with instance separation and per-cell measurements. FastAPI/Gradio, W&B experiment tracking.
