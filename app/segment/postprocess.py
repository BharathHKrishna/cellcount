"""Instance separation (watershed / connected components) and per-cell measurements."""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential, watershed


def separate_instances(binary_mask, min_size=15, min_distance=5, seed_threshold=2):
    """Split a binary foreground mask into individual touching-cell instances.

    binary_mask: numpy array (H, W) of 0/1.
    min_size: components/instances smaller than this (in px) are dropped as
        prediction noise rather than real cells (matches Cellpose's own
        min_size=15 default, for apples-to-apples counts).
    seed_threshold: minimum distance-transform value (roughly, distance to
        the nearest edge) for a watershed seed. Suppresses shallow noise
        bumps that would otherwise shatter one blob into many tiny fake
        instances -- but a blob thinner than this everywhere still gets
        exactly one fallback seed at its deepest point, so real small/thin
        cells aren't dropped outright (see the seed_threshold=2 vs. no
        fallback difference: dropping small isolated blobs *increases*
        count error and hurts Dice, it doesn't just suppress noise).
    Returns: numpy array (H, W) of integer instance labels (0 = background).
    """
    binary_mask = remove_small_objects(np.asarray(binary_mask).astype(bool), max_size=min_size - 1)
    distance = ndi.distance_transform_edt(binary_mask)

    coords = peak_local_max(distance, labels=binary_mask, min_distance=min_distance, threshold_abs=seed_threshold)
    seed_mask = np.zeros_like(distance, dtype=bool)
    seed_mask[tuple(coords.T)] = True

    components, n_components = ndi.label(binary_mask)
    seeded_components = set(components[seed_mask])
    for component_id in range(1, n_components + 1):
        if component_id in seeded_components:
            continue
        component_mask = components == component_id
        peak_flat = np.argmax(np.where(component_mask, distance, -1))
        seed_mask.flat[peak_flat] = True

    markers = ndi.label(seed_mask)[0]
    labels = watershed(-distance, markers, mask=binary_mask)
    labels = remove_small_objects(labels, max_size=min_size - 1)
    return relabel_sequential(labels)[0]


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
