"""Draw outlines + labels for a segmented image."""


def draw_outlines(image, instance_mask):
    """Overlay instance boundaries (and optionally labels) on the original image.

    image: numpy array (H, W) or (H, W, C).
    instance_mask: numpy array (H, W) of integer instance labels (0 = background).
    Returns: numpy array (H, W, 3) — image with outlines drawn on top.
    """
    raise NotImplementedError("skimage.segmentation.find_boundaries per-instance, "
                               "draw over a copy of image with cv2 or matplotlib")
