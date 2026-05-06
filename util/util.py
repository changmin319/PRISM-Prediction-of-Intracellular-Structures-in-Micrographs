from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    """
    Convert a PyTorch tensor into a NumPy image array.

    - single channel `[H, W]`
    - rgb 3 channels `[H, W, 3]`
    """
    image_numpy = image_tensor.cpu().float().numpy()

    # [-1,1] -> [0,255]
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = np.round(image_numpy).astype(imtype)

    if image_numpy.shape[-1] == 1:
        image_numpy = image_numpy[:, :, 0]

    return image_numpy


# Converts a Tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()

    # if label_tensor.dim() == 4:
    #     label_tensor = label_tensor.squeeze(0)  # Convert [1, C, H, W] to [C, H, W]

    if n_label == 0:
        label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0)) * 255.0
        return label_numpy.astype(imtype)


    # if n_label == 0:
    #     label_tensor = (label_tensor + 1) / 2.0  # Normalize to [0, 1]
    #     label_numpy = (label_tensor.numpy() * 255).clip(0, 255).transpose(1, 2, 0)
    #     return label_numpy.astype(imtype)

    unique_labels = torch.unique(label_tensor)
    print(f"Unique labels in tensor before colorization: {unique_labels}")

    assert unique_labels.max() < n_label, \
        f"Label value {unique_labels.max()} exceeds colormap size {n_label}. Check your label preprocessing."

    if label_tensor.size(0) > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]

    colorizer = Colorize(n_label)
    label_tensor = colorizer(label_tensor)

    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    print(f"Visual label shape: {label_numpy.shape}, sample values: {np.unique(label_numpy)}")

    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    if image_numpy.ndim == 3 and image_numpy.shape[-1] == 1:
        image_numpy = image_numpy[:, :, 0]
    elif image_numpy.ndim == 4:
        image_numpy = image_numpy[0]
    elif image_numpy.ndim != 2 and image_numpy.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image_numpy.shape}")

    print(f"Saving image with shape: {image_numpy.shape}, dtype: {image_numpy.dtype}")
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    """
    Create multiple directories if they do not exist.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """
    Create a single directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def uint82bin(n, count=8):
    """Returns the binary representation of integer n, count refers to bit count."""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
