"""v1 pretrained segmentation via Cellpose."""


def run_cellpose(image, model_type="cyto"):
    """Run pretrained Cellpose on an image and return an instance-labeled mask.

    image: numpy array (H, W) or (H, W, C).
    model_type: "cyto" for whole cells, "nuclei" for nucleus-only images.
    Returns: numpy array (H, W) of integer instance labels (0 = background).
    """
    raise NotImplementedError("call cellpose.models.Cellpose(model_type=model_type).eval(image)")
