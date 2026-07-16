"""v1: upload -> segment -> overlay + count, served as a Gradio app."""

import gradio as gr

from segment.cellpose_runner import run_cellpose
from segment.postprocess import measure_instances
from segment.overlay import draw_outlines


def segment_image(image):
    masks = run_cellpose(image)
    measurements = measure_instances(masks)
    overlay = draw_outlines(image, masks)
    return overlay, f"{len(measurements)} cells detected"


demo = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(type="numpy", label="Microscopy image"),
    outputs=[gr.Image(label="Segmented"), gr.Textbox(label="Count")],
    title="CellCount",
    description="Upload a microscopy image to outline and count individual cells.",
)

if __name__ == "__main__":
    demo.launch()
