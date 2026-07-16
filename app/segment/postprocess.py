"""Instance separation (watershed / connected components) and per-cell measurements."""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


def separate_instances(binary_mask):
    """Split a binary foreground mask into individual touching-cell instances.

    binary_mask: numpy array (H, W) of 0/1.
    Returns: numpy array (H, W) of integer instance labels (0 = background).
    """
    binary_mask = np.asarray(binary_mask).astype(bool)
    distance = ndi.distance_transform_edt(binary_mask)
    coords = peak_local_max(distance, labels=binary_mask, min_distance=5)
    peak_mask = np.zeros_like(distance, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers = ndi.label(peak_mask)[0]
    return watershed(-distance, markers, mask=binary_mask)


def measure_instances(instance_mask):
    """Compute per-cell measurements (area, centroid, ...) from a labeled mask.

    instance_mask: numpy array (H, W) of integer instance labels (0 = background).
    Returns: list of dicts, one per instance, e.g. {"label": 1, "area_px": 342, "centroid": (y, x)}.
    """
    return [
        {
            "label": int(prop.label),
            "area_px": int(prop.area),
            "centroid": (float(prop.centroid[0]), float(prop.centroid[1])),
        }
        for prop in regionprops(np.asarray(instance_mask))
    ]
