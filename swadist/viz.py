import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image


def show_imgs(imgs, suptitle=None, titles=None, figheight=5):
    """Plot a grid of images

    Adapted from https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py

    Parameters
    ----------
    imgs: Union[Tensor, List[Tensor]]
    suptitle: str
    titles: List[str]
    figheight: int
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False,
                            figsize=(1.25*len(imgs)*figheight, figheight))
    if suptitle is not None:
        fig.suptitle(suptitle)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if titles is not None:
            axs[0, i].title.set_text(titles[i])
