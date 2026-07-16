"""v2: your own trained U-Net, as an alternative to the Cellpose baseline."""


def run_unet(image, weights_path="models/unet_v1.pt"):
    """Run the trained U-Net on an image and return an instance-labeled mask.

    image: numpy array (H, W) or (H, W, C).
    Returns: numpy array (H, W) of integer instance labels (0 = background),
    after watershed/connected-components instance separation on the raw mask.
    """
    raise NotImplementedError("load unet weights, predict a binary/probability mask, "
                               "then separate touching instances (see postprocess.py)")
