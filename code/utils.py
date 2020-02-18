import numpy as np
from matplotlib import colors
import matplotlib.patches as mpatches

import torch


def convert_to_np(tensor):
    # convert pytorch tensors to numpy arrays
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor


def labels_to_dfc(tensor, no_savanna):
    """
    INPUT:
    Classes encoded in the training scheme (0-9 if savanna is a valid label
    or 0-8 if not). Invalid labels are marked by 255 and will not be changed.

    OUTPUT:
    Classes encoded in the DFC2020 scheme (1-10, and 255 for invalid).
    """

    # transform to numpy array
    tensor = convert_to_np(tensor)

    # copy the original input
    out = np.copy(tensor)

    # shift labels if there is no savanna class
    if no_savanna:
        for i in range(2, 9):
            out[tensor == i] = i + 1
    else:
        pass

    # transform from zero-based labels to 1-10
    out[tensor != 255] += 1

    # make sure the mask is intact and return transformed labels
    assert np.all((tensor == 255) == (out == 255))
    return out


def display_input_batch(tensor, display_indices=0, brightness_factor=3):

    # extract display channels
    tensor = tensor[:, display_indices, :, :]

    # restore NCHW tensor shape if single channel image
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)

    # scale image
    tensor = torch.clamp((tensor * brightness_factor), 0, 1)

    return tensor


def display_label_batch(tensor, no_savanna=False):

    # get predictions if input is one-hot encoded
    if len(tensor.shape) == 4:
        tensor = tensor.max(1)[1]

    # convert train labels to DFC2020 class scheme
    tensor = labels_to_dfc(tensor, no_savanna)

    # colorize labels
    cmap = mycmap()
    imgs = []
    for s in range(tensor.shape[0]):
        im = (tensor[s, :, :] - 1) / 10
        im = cmap(im)[:, :, 0:3]
        im = np.rollaxis(im, 2, 0)
        imgs.append(im)
    tensor = np.array(imgs)

    return tensor


def classnames():
    return ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
            "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"]


def mycmap():
    cmap = colors.ListedColormap(['#009900',
                                  '#c6b044',
                                  '#fbff13',
                                  '#b6ff05',
                                  '#27ff87',
                                  '#c24f44',
                                  '#a5a5a5',
                                  '#69fff8',
                                  '#f9ffa4',
                                  '#1c0dff',
                                  '#ffffff'])
    return cmap


def mypatches():
    patches = []
    for counter, name in enumerate(classnames()):
        patches.append(mpatches.Patch(color=mycmap().colors[counter],
                                      label=name))
    return patches
