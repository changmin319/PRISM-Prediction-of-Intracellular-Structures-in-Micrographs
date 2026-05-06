import os.path
import zarr
import random
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
from data.tiles_visualization import (
    visualize_background_only,
    visualize_train_test,
    visualize_patches_on_image,
    visualize_patches_on_image_with_index,
)
import json
import os
from PIL import ImageEnhance
from torch.utils.data import Sampler
import itertools

# Utility
def normalize8(img):
    img = img.astype(np.float32)
    vmin = float(img.min())
    vmax = float(img.max())
    rng = vmax - vmin
    if rng <= 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - vmin) / rng * 255.0
    return img.astype(np.uint8)

def to_tensor_gray(pil_img):
    t = torch.unsqueeze(torch.Tensor(np.array(pil_img)) / 255.0, dim=0)
    t = (t - 0.5) / 0.5
    return t

def shuffle_pool(lst):
    shuffled = lst.copy()
    random.shuffle(shuffled)
    return shuffled

class BalancedBatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_pos, batch_neg):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_pos = batch_pos
        self.batch_neg = batch_neg
        self.pos_pool = []
        self.neg_pool = []

    def __iter__(self):
        while True:
            if len(self.pos_pool) < self.batch_pos:
                self.pos_pool = self.pos_indices.copy()
                random.shuffle(self.pos_pool)
            if len(self.neg_pool) < self.batch_neg:
                self.neg_pool = self.neg_indices.copy()
                random.shuffle(self.neg_pool)
            batch = [self.pos_pool.pop() for _ in range(self.batch_pos)] + \
                    [self.neg_pool.pop() for _ in range(self.batch_neg)]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return (len(self.pos_indices) + len(self.neg_indices)) // (self.batch_pos + self.batch_neg)

def _apply_intensity(img, intensity_aug_prob=0.5):
    if random.random() < intensity_aug_prob:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    if random.random() < intensity_aug_prob:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img

def intensity_aware_crop(image, mask, crop_size, max_attempts=10, intensity_threshold=0.2):

    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    h, w = image.shape[:2]

    for _ in range(max_attempts):
        y = np.random.randint(0, max(1, h - crop_size + 1)) if h > crop_size else 0
        x = np.random.randint(0, max(1, w - crop_size + 1)) if w > crop_size else 0

        crop_mask = mask[y:y+crop_size, x:x+crop_size]
        crop_mask_norm = crop_mask / 255.0 if crop_mask.max() > 1 else crop_mask

        if crop_mask_norm.max() > intensity_threshold:
            crop_image = image[y:y+crop_size, x:x+crop_size]
            return crop_image, crop_mask, (x, y)

    # fallback: center crop
    cy = max(0, (h - crop_size) // 2)
    cx = max(0, (w - crop_size) // 2)
    crop_image = image[cy:cy+crop_size, cx:cx+crop_size]
    crop_mask = mask[cy:cy+crop_size, cx:cx+crop_size]
    return crop_image, crop_mask, (cx, cy)


def random_crop(image, mask, crop_size):

    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    h, w = image.shape[:2]
    
    if h > crop_size and w > crop_size:
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)
    else:
        y = x = 0
    
    crop_image = image[y:y+crop_size, x:x+crop_size]
    crop_mask = mask[y:y+crop_size, x:x+crop_size]
    
    return crop_image, crop_mask, (x, y)


class OmeZarrPatchDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt

        zarr_path_1 = os.path.join(opt.data_path_1, opt.dataset_name_1)
        if zarr_path_1.startswith('file:///'):
            zarr_path_1 = zarr_path_1.replace('file:///', '')
        store_1 = zarr.open(zarr_path_1, mode="r")
        data_1 = store_1["0"]
        haadf_full_1 = data_1[opt.haadf_channel_1 if opt.haadf_channel_1 >= 0 else 0]
        height_1, width_1 = haadf_full_1.shape
        patch_size = opt.patch_size
        overlap = opt.overlap
        stride = int(patch_size * (1 - overlap))
        splits_1 = []
        for y_top in range(0, height_1 - patch_size + 1, stride):
            for x_left in range(0, width_1 - patch_size + 1, stride):
                splits_1.append([y_top, x_left])
        total_tiles_1 = 132

        preset = (getattr(self.opt, "structure_preset", "") or "").strip()
        if not preset:
            preset = _ACTIVE_STRUCTURE_PRESET
        if preset not in _STRUCTURE_TILE_PRESETS:
            keys = ", ".join(sorted(_STRUCTURE_TILE_PRESETS))
            raise ValueError(f"Unknown structure_preset={preset!r}. Use one of: {keys}")
        train_positive_indices_1, test_positive_indices_1 = _STRUCTURE_TILE_PRESETS[preset]
        print(f"[INFO] Tile positive preset: {preset} (132-tile EM; see _STRUCTURE_TILE_PRESETS / _ACTIVE_STRUCTURE_PRESET at end of omezarr_patch_dataset.py)")

        all_indices_1 = set(range(total_tiles_1))
        positive_indices_1 = set(train_positive_indices_1 + test_positive_indices_1)
        negative_candidates_1 = list(all_indices_1 - positive_indices_1)

        random.seed(42)

        train_negative_indices_1 = random.sample(negative_candidates_1, 60)

        remaining_negatives = list(set(negative_candidates_1) - set(train_negative_indices_1))

        num_test_negatives = len(test_positive_indices_1)

        test_negative_indices_1 = random.sample(
            remaining_negatives,
            min(num_test_negatives, len(remaining_negatives))
        )

        opt.manual_train_indices_1 = train_positive_indices_1 + train_negative_indices_1
        opt.manual_test_indices_1 = test_positive_indices_1 + test_negative_indices_1

        split_info = {"train_positive_indices_1": train_positive_indices_1,
                      "test_positive_indices_1": test_positive_indices_1,
                      "train_negative_indices_1": train_negative_indices_1,
                      "test_negative_indices_1": test_negative_indices_1, "splits_1": splits_1}

        split_path = os.path.join(os.path.dirname(__file__), f"tile_split_indices_{preset}.json")

        with open(split_path, "w") as f:
            json.dump(split_info, f)

        self.train_pos_indices_1 = train_positive_indices_1
        self.train_neg_indices_1 = train_negative_indices_1

        dataset_config = {
            'path': opt.data_path_1,
            'name': opt.dataset_name_1,
            'haadf_channel': opt.haadf_channel_1,
            'target_channel': opt.target_channel_1,
            'manual_train_indices': opt.manual_train_indices_1,
            'manual_test_indices': opt.manual_test_indices_1
        }

        self.datasets = []
        manual_train_indices = []
        if dataset_config['manual_train_indices']:
            if isinstance(dataset_config['manual_train_indices'], str):
                manual_train_indices = [int(x) for x in dataset_config['manual_train_indices'].split(',')]
            else:
                manual_train_indices = list(dataset_config['manual_train_indices'])

        zarr_path = os.path.join(dataset_config['path'], dataset_config['name'])
        print(f"Loading dataset: {zarr_path}")
        self.opt.train_pos_indices_current = self.train_pos_indices_1
        self.opt.train_neg_indices_current = self.train_neg_indices_1
        dataset = self._load_single_dataset(
            zarr_path,
            f"dataset_1",
            dataset_config['haadf_channel'],
            dataset_config['target_channel'],
            manual_train_indices,
            dataset_config.get('manual_test_indices', None)
        )
        self.datasets.append(dataset)
        if not self.datasets:
            raise ValueError("No valid dataset found. Please check data_path_1 and dataset_name_1 parameters.")

        self.patch_size = self.opt.patch_size  # 512

        if not self.opt.isTrain:

            self.sample_list = [(0, local_idx) for local_idx in range(len(self.datasets[0]['splits']))]
            print(f"[TEST] total test tiles: {len(self.sample_list)}")
            return


        d0 = self.datasets[0]
        self.pos_local_indices = list(d0['pos_local_indices'])
        self.neg_local_indices = list(d0['neg_local_indices'])

        self.target_pos_samples = getattr(self.opt, 'target_pos_samples', 2500)
        self.target_neg_samples = getattr(self.opt, 'target_neg_samples', 2500)
        self.total_train_samples = self.target_pos_samples + self.target_neg_samples


        self.intensity_threshold = getattr(self.opt, 'intensity_threshold', getattr(self.opt, 'insulin_threshold', 0.2))
        self.max_crop_attempts = getattr(self.opt, 'max_crop_attempts', 10)

        #
        self.rot_angles = [0, 90, 180, 270]

        print(f"[INFO] Online augmentation enabled.")
        print(f"[INFO] Positive tiles: {len(self.pos_local_indices)}")
        print(f"[INFO] Negative tiles: {len(self.neg_local_indices)}")
        print(f"[INFO] Epoch samples: pos={self.target_pos_samples}, neg={self.target_neg_samples}, total={self.total_train_samples}")
        print(f"[INFO] Intensity-aware crop threshold (structure map): {self.intensity_threshold}, max attempts: {self.max_crop_attempts}")

    def _load_single_dataset(self, zarr_path, dataset_name, haadf_channel, target_channel, manual_train_indices=None, manual_test_indices=None):
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Dataset not found: {zarr_path}")

        store = zarr.open(zarr_path, mode="r")
        data = store["0"]  # (C, H, W)

        patch_size = self.opt.patch_size
        overlap = self.opt.overlap
        keep_ratio = getattr(self.opt, "keep_ratio", 0.0)

        haadf_full = data[haadf_channel]
        valid_mask = haadf_full > 0

        stride = int(patch_size * (1 - overlap))
        height, width = haadf_full.shape

        all_possible_tiles = []
        for y_top in range(0, height - patch_size + 1, stride):
            for x_left in range(0, width - patch_size + 1, stride):
                all_possible_tiles.append((y_top, x_left))

        splits = []
        orig_indices = []
        for orig_idx, (y_top, x_left) in enumerate(all_possible_tiles):
            y_bot = y_top + patch_size
            x_right = x_left + patch_size
            target_patch = data[target_channel, y_top:y_bot, x_left:x_right]
            if (target_patch > 0).mean() < self.opt.target_presence_threshold:
                continue
            if valid_mask[y_top:y_bot, x_left:x_right].mean() >= keep_ratio:
                splits.append((y_top, x_left))
                orig_indices.append(orig_idx)

        if manual_train_indices is None: manual_train_indices = []
        if manual_test_indices is None: manual_test_indices = []

        if manual_train_indices and manual_test_indices:
            splits_train = [splits[i] for i in manual_train_indices if i < len(splits)]
            splits_test = [splits[i] for i in manual_test_indices if i < len(splits)]
        elif manual_train_indices:
            splits_train = [splits[i] for i in manual_train_indices if i < len(splits)]
            splits_test = [p for i, p in enumerate(splits) if i not in manual_train_indices]
        elif manual_test_indices:
            splits_test = [splits[i] for i in manual_test_indices if i < len(splits)]
            splits_train = [p for i, p in enumerate(splits) if i not in manual_test_indices]
        else:
            splits_train = []
            splits_test = splits

        if self.opt.isTrain:
            local2orig = {k: manual_train_indices[k] for k in range(len(manual_train_indices))}
        else:
            local2orig = {k: manual_test_indices[k] for k in range(len(manual_test_indices))}

        print(f"\nVisualization: {dataset_name}")
        haadf_8bit = normalize8(haadf_full)
        target_full = data[target_channel]
        target_8bit = normalize8(target_full)
        print(f"Total patches: {len(splits)}")
        print(f"Training patches: {len(splits_train)}")
        print(f"Test patches: {len(splits_test)}")
        print(f"Manual train indices: {manual_train_indices}")

        try:
            print("Displaying split visualization...")
            # visualize_background_only(haadf_8bit, title=f"{dataset_name} - HAADF Full")
            # visualize_background_only(target_8bit, title=f"{dataset_name} - Structure map (full)")

            if len(splits) > 0:
                test_indices = []
                if len(splits_test) > 0:
                    test_coords_set = set(splits_test)
                    for i, coord in enumerate(splits):
                        if coord in test_coords_set:
                            test_indices.append(i)

                print(f"Visualizing {len(splits)} patches ({len(test_indices)} test) on HAADF...")
                # visualize_patches_on_image_with_index(
                #     image=haadf_8bit,
                #     coords=splits,
                #     test_indices=test_indices,
                #     patch_size=patch_size,
                #     title=f"{dataset_name} - HAADF Patches (Red:Train, Cyan:Test)",
                #     force_gray = True
                # )
            # #
                print(f"Visualizing {len(splits)} patches ({len(test_indices)} test) on structure map...")
                # visualize_patches_on_image_with_index(
                #     image=target_8bit,
                #     coords=splits,
                #     test_indices=test_indices,
                #     patch_size=patch_size,
                #     title=f"{dataset_name} - Structure map patches (red: train, cyan: test)",
                #     force_gray=True
                # )

            # else:
            #     print("[VISUALIZATION] No patches to visualize.")

        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Continuing without visualization...")

        if self.opt.isTrain:
            orig2local = {orig: k for k, orig in enumerate(manual_train_indices)}
            pos_local  = [orig2local[i] for i in self.opt.train_pos_indices_current if i in orig2local]
            neg_local  = [orig2local[i] for i in self.opt.train_neg_indices_current if i in orig2local]
        else:
            orig2local = {}
            pos_local = neg_local = []

        return {
            'name': dataset_name,
            'data': data,
            'splits': splits_train if self.opt.isTrain else splits_test,
            'haadf_channel': haadf_channel,
            'target_channel': target_channel,
            'patch_size': patch_size,
            'pos_local_indices': pos_local,
            'neg_local_indices': neg_local,
            'local2orig': local2orig
        }

    def _random_aug_params(self):
        angle = random.choice(self.rot_angles)
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5
        do_intensity = random.random() < 0.5

        strategy = random.random()
        if strategy < 0.2:
            scale_area = random.uniform(0.7, 0.8)
        elif strategy < 0.5:
            scale_area = random.uniform(0.8, 0.95)
        else:
            scale_area = random.uniform(0.95, 1.0)

        side_ratio = np.sqrt(scale_area)
        crop_size = max(1, int(self.patch_size * side_ratio))
        return angle, hflip, vflip, dflip, do_intensity, crop_size

    def __getitem__(self, index):
        if not self.opt.isTrain:
            _, local_idx = self.sample_list[index]
            dataset = self.datasets[0]
            y_top, x_left = dataset['splits'][local_idx]
            orig_idx = dataset['local2orig'][local_idx]
            patch = dataset['data'][:, y_top:y_top + dataset['patch_size'], x_left:x_left + dataset['patch_size']]
            haadf = Image.fromarray(normalize8(patch[dataset['haadf_channel']]))
            target = Image.fromarray(normalize8(patch[dataset['target_channel']]))
            haadf_tensor = to_tensor_gray(haadf)
            target_tensor = to_tensor_gray(target)
            return {
                "haadf": haadf_tensor,
                "structuremap": target_tensor,
                "path": f"{dataset['name']}_Patch({y_top},{x_left})",
                "dataset_idx": 0,
                "tile_coord": (y_top, x_left),
                "orig_idx": orig_idx,
            }
        else:

            is_pos = index < self.target_pos_samples
            dataset = self.datasets[0]


            if is_pos:
                base_idx = random.choice(self.pos_local_indices)
            else:
                base_idx = random.choice(self.neg_local_indices)

            y_top, x_left = dataset['splits'][base_idx]


            patch = dataset['data'][:, y_top:y_top + dataset['patch_size'], x_left:x_left + dataset['patch_size']]
            haadf = Image.fromarray(normalize8(patch[dataset['haadf_channel']]))
            target = Image.fromarray(normalize8(patch[dataset['target_channel']]))


            angle, hflip, vflip, dflip, do_intensity, crop_size = self._random_aug_params()
            
            if is_pos:
                # Positive: intensity-aware crop (prefer windows with sufficient structure-map signal)
                haadf_cropped, target_cropped, (crop_x, crop_y) = intensity_aware_crop(
                    haadf, target, crop_size,
                    max_attempts=self.max_crop_attempts,
                    intensity_threshold=self.intensity_threshold
                )
            else:
                # Negative
                haadf_cropped, target_cropped, (crop_x, crop_y) = random_crop(
                    haadf, target, crop_size
                )

            x_left_aug = x_left + crop_x
            y_top_aug  = y_top  + crop_y

            if isinstance(haadf_cropped, np.ndarray):
                haadf_cropped = Image.fromarray(haadf_cropped)
            if isinstance(target_cropped, np.ndarray):
                target_cropped = Image.fromarray(target_cropped)


            haadf = haadf_cropped.resize((self.patch_size, self.patch_size), Image.LANCZOS)
            target = target_cropped.resize((self.patch_size, self.patch_size), Image.NEAREST)

            if hflip:
                haadf = haadf.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            if vflip:
                haadf = haadf.transpose(Image.FLIP_TOP_BOTTOM)
                target = target.transpose(Image.FLIP_TOP_BOTTOM)
            if dflip:
                haadf = haadf.transpose(Image.TRANSPOSE)
                target = target.transpose(Image.TRANSPOSE)

            if angle != 0:
                haadf = haadf.rotate(angle, resample=Image.BILINEAR, expand=False)
                target = target.rotate(angle, resample=Image.NEAREST,  expand=False)


            if do_intensity:
                haadf = _apply_intensity(haadf, getattr(self.opt, 'intensity_aug_prob', 0.5))

            haadf_tensor = to_tensor_gray(haadf)
            target_tensor = to_tensor_gray(target)

            return {
                "haadf": haadf_tensor,
                "structuremap": target_tensor,
                "path": f"{dataset['name']}_Patch({y_top_aug},{x_left_aug}),crop_{crop_size}->{self.patch_size},aug={angle},{hflip},{vflip},{dflip},{do_intensity}",
                "dataset_idx": 0,
                "tile_coord": (y_top_aug, x_left_aug),
                "angle": angle,
                "hflip": hflip,
                "vflip": vflip,
                "dflip": dflip,
                "intensity": do_intensity,
                "crop_type": f'crop_{crop_size}',
                "is_positive": is_pos
            }

    def __len__(self):
        if not self.opt.isTrain:
            return len(self.sample_list)
        return getattr(self, 'total_train_samples', 0) or (len(self.pos_local_indices) + len(self.neg_local_indices))

    def name(self):
        return 'OmeZarrPatchDataset'

    def is_positive(self, index):
        if not self.opt.isTrain:
            return False
        return index < self.target_pos_samples


# Same EM: 132-tile grid. Only train/test positive indices differ by structure (channel / label).
# Switch runs: change _ACTIVE_STRUCTURE_PRESET, or pass --structure_preset <key> (TrainOptions).
_ACTIVE_STRUCTURE_PRESET = "insulin"

_STRUCTURE_TILE_PRESETS = {
    "nucleic_acids": (
        [1, 2, 11, 20, 25, 27, 36, 37, 38, 39,
         16, 17, 44, 45, 47, 49, 50, 60, 61, 67,
         70, 71, 72, 73, 74, 81, 82, 83, 92, 93, 96,
         97, 104, 106, 105, 107],
        [3, 26, 48, 51, 66, 84, 94, 118, 131],
    ),
    "zymogen": (
        [0, 1, 3, 23, 24, 45, 46, 47, 67, 114],
        [2, 22, 25],
    ),
    "insulin": (
        [8, 9, 10, 11, 12, 13, 21, 34, 35, 53, 55, 57,
         58, 59, 75, 76, 77, 78, 79, 98, 99, 101, 102,
         109, 111, 112, 125],
        [31, 56, 54, 80, 100, 110],
    ),
    "lysosomes": (
        [4, 6, 7, 13, 14, 25, 26, 27, 35, 39, 40,
         45, 46, 47, 51, 59, 60, 61, 63, 66, 69,
         76, 77, 85, 89, 99, 100, 107, 115, 119,
         120, 121, 122, 128, 129, 131, 20, 21],
        [5, 8, 33, 62, 68, 71, 78, 88, 114, 43],
    ),
    "glucagon": (
        [9, 10, 17, 30, 32, 33, 38, 39, 53, 54,
         55, 59, 61, 83, 84, 85, 102, 103, 105,
         107, 120, 125, 126, 127, 128],
        [8, 11, 31, 75, 81, 86, 104],
    ),
}
