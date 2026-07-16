"""v1 pretrained segmentation via Cellpose.

Cellpose 4.x ships a single generalist model ("cpsam", Cellpose-SAM) that
replaced the old per-tissue "cyto"/"nuclei" checkpoints from Cellpose <=2.x.
`model_type` is kept for API stability / forward-compat with the model zoo
(cellpose.models.MODEL_NAMES) but the default "cpsam_v2" handles both whole
cells and nuclei out of the box.
"""

_MODEL_CACHE = {}


def _get_model(model_type):
    from cellpose import models

    if model_type not in _MODEL_CACHE:
        _MODEL_CACHE[model_type] = models.CellposeModel(pretrained_model=model_type)
    return _MODEL_CACHE[model_type]


def run_cellpose(image, model_type="cpsam_v2"):
    """Run pretrained Cellpose on an image and return an instance-labeled mask.

    image: numpy array (H, W) or (H, W, C).
    model_type: name of a Cellpose model checkpoint (see cellpose.models.MODEL_NAMES).
    Returns: numpy array (H, W) of integer instance labels (0 = background).
    """
    model = _get_model(model_type)
    masks, _flows, _styles = model.eval(image, diameter=None)
    return masks
