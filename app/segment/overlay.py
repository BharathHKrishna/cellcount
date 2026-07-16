"""Draw outlines + labels for a segmented image."""

import cv2
import numpy as np
from skimage.segmentation import find_boundaries


def draw_outlines(image, instance_mask, color=(255, 0, 0), label_cells=True):
    """Overlay instance boundaries (and optionally labels) on the original image.

    image: numpy array (H, W) or (H, W, C).
    instance_mask: numpy array (H, W) of integer instance labels (0 = background).
    Returns: numpy array (H, W, 3) — image with outlines drawn on top.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        base = cv2.cvtColor(_to_uint8(image), cv2.COLOR_GRAY2RGB)
    else:
        base = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGBA2RGB) if image.shape[-1] == 4 else _to_uint8(image).copy()

    boundaries = find_boundaries(instance_mask, mode="outer")
    base[boundaries] = color

    if label_cells:
        for prop_label, centroid in _centroids(instance_mask):
            y, x = int(centroid[0]), int(centroid[1])
            cv2.putText(base, str(prop_label), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 0), 1, cv2.LINE_AA)

    return base


def _to_uint8(image):
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float64)
    image -= image.min()
    max_val = image.max()
    if max_val > 0:
        image /= max_val
    return (image * 255).astype(np.uint8)


def _centroids(instance_mask):
    from skimage.measure import regionprops
    return [(prop.label, prop.centroid) for prop in regionprops(np.asarray(instance_mask))]
