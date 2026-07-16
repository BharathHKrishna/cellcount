"""Instance separation (watershed / connected components) and per-cell measurements."""


def separate_instances(binary_mask):
    """Split a binary foreground mask into individual touching-cell instances.

    binary_mask: numpy array (H, W) of 0/1.
    Returns: numpy array (H, W) of integer instance labels (0 = background).
    """
    raise NotImplementedError("skimage.segmentation.watershed on the distance transform, "
                               "seeded from local maxima / peak_local_max")


def measure_instances(instance_mask):
    """Compute per-cell measurements (area, centroid, ...) from a labeled mask.

    instance_mask: numpy array (H, W) of integer instance labels (0 = background).
    Returns: list of dicts, one per instance, e.g. {"label": 1, "area_px": 342, "centroid": (y, x)}.
    """
    raise NotImplementedError("skimage.measure.regionprops on instance_mask")
