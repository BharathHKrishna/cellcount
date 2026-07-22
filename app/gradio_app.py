"""Upload -> segment -> overlay + count + measurements, served as a Gradio app.

Supports toggling between the pretrained Cellpose baseline and the
project's own trained U-Net (once models/unet_v1.pt exists), with a
side-by-side comparison view and downloadable CSV of per-cell measurements.

Set CELLCOUNT_LITE=1 to run U-Net-only (no Cellpose import, no SAM weight
download) -- used for the public deploy, since Cellpose's SAM backbone
needs ~1.2GB of RAM that free hosting tiers don't reliably provide.
"""

import glob
import os

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment.overlay import draw_outlines
from segment.postprocess import measure_instances

LITE_MODE = os.environ.get("CELLCOUNT_LITE") == "1"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
UNET_WEIGHTS = os.path.join(MODELS_DIR, "unet_v1.pt")
UNET_AVAILABLE = os.path.exists(UNET_WEIGHTS)
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

MODEL_CHOICES = ["My U-Net"] if LITE_MODE else (
    ["Cellpose (pretrained)", "My U-Net"] if UNET_AVAILABLE else ["Cellpose (pretrained)"]
)
DEFAULT_MODEL = "My U-Net" if LITE_MODE else MODEL_CHOICES[0]


def _run_model(image, model_choice):
    if model_choice == "My U-Net" and UNET_AVAILABLE:
        from segment.unet_model import run_unet
        return run_unet(image, weights_path=UNET_WEIGHTS)
    from segment.cellpose_runner import run_cellpose
    return run_cellpose(image)


def _size_histogram(measurements):
    areas = [m["area_px"] for m in measurements]
    fig, ax = plt.subplots(figsize=(4, 3))
    if areas:
        ax.hist(areas, bins=min(30, max(5, len(areas) // 3)), color="#4C72B0")
    ax.set_xlabel("Cell area (px)")
    ax.set_ylabel("Count")
    ax.set_title("Size distribution")
    fig.tight_layout()
    return fig


def _to_csv(measurements):
    df = pd.DataFrame(measurements)
    path = os.path.join(os.path.dirname(__file__), "_last_measurements.csv")
    df.to_csv(path, index=False)
    return path


def segment_image(image, model_choice):
    if image is None:
        return None, "Upload an image first.", None, None

    masks = _run_model(image, model_choice)
    measurements = measure_instances(masks)
    overlay = draw_outlines(image, masks)
    csv_path = _to_csv(measurements)
    hist = _size_histogram(measurements)
    return overlay, f"{len(measurements)} cells detected", hist, csv_path


def compare_models(image):
    if image is None:
        return None, None, "Upload an image first.", "Upload an image first."

    from segment.cellpose_runner import run_cellpose
    cellpose_masks = run_cellpose(image)
    cellpose_measurements = measure_instances(cellpose_masks)
    cellpose_overlay = draw_outlines(image, cellpose_masks)
    cellpose_count = f"Cellpose: {len(cellpose_measurements)} cells"

    if not UNET_AVAILABLE:
        return cellpose_overlay, None, cellpose_count, "My U-Net: not trained yet (models/unet_v1.pt missing)"

    from segment.unet_model import run_unet
    unet_masks = run_unet(image, weights_path=UNET_WEIGHTS)
    unet_measurements = measure_instances(unet_masks)
    unet_overlay = draw_outlines(image, unet_masks)
    unet_count = f"My U-Net: {len(unet_measurements)} cells"

    return cellpose_overlay, unet_overlay, cellpose_count, unet_count


with gr.Blocks(title="CellCount") as demo:
    gr.Markdown("# CellCount\nUpload a microscopy image to outline and count individual cells.")
    if LITE_MODE:
        gr.Markdown(
            "_Running in lite mode: only the project's own U-Net is available here. "
            "Cellpose's SAM backbone needs ~1.2GB RAM that this free deployment doesn't provide "
            "— run the app locally (see README) for the full Cellpose-vs-U-Net comparison._"
        )

    with gr.Tab("Segment"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Microscopy image")
                gr.Examples(
                    examples=[[p] for p in sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.png")))],
                    inputs=[image_input],
                    label="No image handy? Try one of these nuclei micrographs (BBBC038/DSB2018)",
                )
                model_choice = gr.Radio(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model")
                run_button = gr.Button("Segment", variant="primary")
            with gr.Column():
                overlay_output = gr.Image(label="Segmented")
                count_output = gr.Textbox(label="Count")
                hist_output = gr.Plot(label="Size distribution")
                csv_output = gr.File(label="Download measurements (CSV)")

        run_button.click(
            segment_image,
            inputs=[image_input, model_choice],
            outputs=[overlay_output, count_output, hist_output, csv_output],
        )

    if not LITE_MODE:
        with gr.Tab("Compare: Cellpose vs My U-Net"):
            with gr.Row():
                compare_input = gr.Image(type="numpy", label="Microscopy image")
            compare_button = gr.Button("Compare", variant="primary")
            with gr.Row():
                with gr.Column():
                    cellpose_overlay_output = gr.Image(label="Cellpose (pretrained)")
                    cellpose_count_output = gr.Textbox(label="Cellpose count")
                with gr.Column():
                    unet_overlay_output = gr.Image(label="My U-Net")
                    unet_count_output = gr.Textbox(label="U-Net count")

            compare_button.click(
                compare_models,
                inputs=[compare_input],
                outputs=[cellpose_overlay_output, unet_overlay_output, cellpose_count_output, unet_count_output],
            )

if __name__ == "__main__":
    demo.launch()
