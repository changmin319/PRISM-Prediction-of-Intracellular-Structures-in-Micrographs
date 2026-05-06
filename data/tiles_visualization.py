import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_background_only(image, title="Structure map"):
    """
    Display a grayscale or RGB image without any patch boxes.
    Accepts numpy array, file path, or PIL Image.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_train_test(full_haadf, train_coords, test_coords, patch_size):
    """
    Show kept patches. Training patches are red, test patches are cyan.
    """
    if len(train_coords) + len(test_coords) == 0:
        print("[WARN] No patch to display.")
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(full_haadf, cmap="gray")
    ax.set_title("HAADF – red: train, cyan: test")
    for y_top, x_left in train_coords:
        rect = patches.Rectangle(
            (x_left, y_top), patch_size, patch_size,
            linewidth=1, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
    for y_top, x_left in test_coords:
        rect = patches.Rectangle(
            (x_left, y_top), patch_size, patch_size,
            linewidth=1, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)
    plt.show()


def visualize_patches_on_image(image, train_coords, test_coords, patch_size, title="Patch Grid"):
    """
    Draw patch boxes over a background image. Train = red, Test = cyan.
    Works for grayscale or RGB images.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title(title)
    for y_top, x_left in train_coords:
        rect = patches.Rectangle(
            (x_left, y_top), patch_size, patch_size,
            linewidth=1, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
    for y_top, x_left in test_coords:
        rect = patches.Rectangle(
            (x_left, y_top), patch_size, patch_size,
            linewidth=1, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_patches_on_image_with_index(image, coords, test_indices=None, patch_size=512, title="Patch Grid", force_gray=False):
    """
    Draw all patch boxes on the background image and label them with their index.
    test_indices: optional, if specified, those patches are drawn in cyan, others in red.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    if force_gray:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)
    ax.set_title(title)

    for idx, (y_top, x_left) in enumerate(coords):
        edgecolor = "cyan" if test_indices and idx in test_indices else "red"
        rect = patches.Rectangle(
            (x_left, y_top), patch_size, patch_size,
            linewidth=1, edgecolor=edgecolor, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x_left + 4, y_top + 14, str(idx),
            color=edgecolor, fontsize=7, weight='bold',
            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2')
        )
    plt.axis('off')
    plt.tight_layout()
    plt.show()
