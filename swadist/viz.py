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


# MAY NOT BE WORKING
def plot_or_log_activations(trainer,
                            loader,
                            img_idx_dict=None,
                            n_imgs=None,
                            save_idxs=False,
                            epoch=None,
                            stage='train',
                            random=False):
    """
    img_idx_dict: dict
        Has entries like { 'img_idxs': [1, 2, 3], 'layer1/conv1': [10, 7, 23], ...}.
    save_idxs: bool
        If True, store the existing or new image indices in this Trainer.
    """
    trainer.model.eval()
    new_idx_dict = False

    if img_idx_dict is None:
        if not random and hasattr(trainer, f'{stage}_img_idx_dict'):
            img_idx_dict = getattr(trainer, f'{stage}_img_idx_dict')
            img_idxs = img_idx_dict['img_idxs']
        else:
            new_idx_dict = True
            img_idx_dict = {}
            n_imgs = 1 if n_imgs is None else n_imgs
            img_idxs = np.random.choice(len(loader.dataset), size=n_imgs, replace=False)
            img_idxs.sort()
            img_idx_dict['img_idxs'] = img_idxs
    else:
        img_idxs = img_idx_dict['img_idxs']

    for i, img_idx in enumerate(img_idxs):
        imgs = []
        image = loader.dataset[img_idx][0]
        original = loader.dataset.inv_transform(image)
        imgs.append(original)
        image = image[None, :]

        for conv in trainer.model.convs:
            out = trainer.model.partial_forward(image, conv)
            if new_idx_dict:
                conv_idx = np.random.choice(out.shape[1], replace=False)
                if not conv in img_idx_dict:
                    img_idx_dict[conv] = [conv_idx]
                else:
                    img_idx_dict[conv].append(conv_idx)
            else:
                conv_idx = img_idx_dict[conv][i]
            imgs.append(out[0, conv_idx, :, :][None, :, :])

        if trainer.logs:
            trainer.writer.add_image(f'{stage}/{img_idx}_0_original', original, epoch)
            for j, conv in enumerate(trainer.model.convs):
                trainer.writer.add_image(f'{stage}/{img_idx}_{j+1}_{conv}', imgs[j+1], epoch)
        else:
            titles = ['original']
            titles.extend(trainer.model.convs)
            suptitle = f'{stage} activations'
            if epoch is not None:
                suptitle += f' after epoch {epoch + 1}'
            show_imgs(imgs, suptitle, titles=titles)

    if save_idxs:
        setattr(trainer, f'{stage}_img_idx_dict', img_idx_dict)

    return img_idx_dict


# MAY NOT BE WORKING
def plot_or_log_filters(trainer, w_idx_dict=None, save_idxs=False,
                        epoch=None, random=False):
    trainer.model.eval()

    if w_idx_dict is None:
        if not random and hasattr(trainer, 'filter_idx_dict'):
            w_idx_dict = trainer.filter_idx_dict
        filters, w_idx_dict = trainer.model.get_filters(w_idx_dict)
    else:
        filters, w_idx_dict = trainer.model.get_filters()

    if trainer.logs:
        for conv, (idxs, filter_group) in filters.items():
            tag = f'{conv}/{idxs}'
            trainer.writer.add_images(tag, filter_group, epoch)
    else:
        imgs = [make_grid(f, nrow=2) for _, (_, f) in filters.items()]
        suptitle = 'filters'
        if epoch is not None:
            suptitle += f' after epoch {epoch + 1}'
        show_imgs(imgs, suptitle, list(w_idx_dict))

    if save_idxs:
        trainer.filter_idx_dict = w_idx_dict

    return w_idx_dict
