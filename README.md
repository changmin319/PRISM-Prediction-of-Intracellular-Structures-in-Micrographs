# PRISM-Prediction-of-Intracellular-Structures-in-Micrographs
**PRISM** (Prediction of Intracellular Structures in Micrographs), a **two-stage pipeline** that uses EDX-derived labels for EM segmentation. A **generative model** is trained to predict EDX-derived structure maps from grayscale EM; the predicted maps are then used to prompt **SAM** for instance segmentation.

PyTorch training and evaluation code built on **pix2pixHD** for **electron microscopy**: conditional image-to-image mapping from **EM** (single channel) to a **structure map** (single channel), using **OME-Zarr** and **patch grids** (e.g. 132 tiles on a given EM).

The high-resolution datasets can be downloaded here: <https://www.nanotomy.org/OA/Tian2026SUB/>

<img width="3601" height="1195" alt="EM-Map-Mask" src="https://github.com/user-attachments/assets/bec72ca9-43fb-4e72-9f17-9ff542012c40" />

## Prerequisites

- **Python 3** (recommended 3.8+)
- **PyTorch** with a matching CUDA build if you use GPU ([pytorch.org](https://pytorch.org))
- **NVIDIA GPU + CUDA** (optional but recommended for training speed)
- **CPU** is supported: pass `--gpu_ids -1` (default in `base_options.py`)

## Installation

1. Install PyTorch (and torchvision) for your platform.

2. Install Python dependencies used across `train.py`, `test.py`, and `data/` (minimum set):

   ```bash
   pip install numpy zarr pillow scipy opencv-python pandas matplotlib torch torchvision
   ```

3. Clone or copy this repo and `cd` into the project root (the folder containing `train.py`).

## Data layout (OME-Zarr)

- Point `--data_path_1` to the **directory that contains** your `.ome.zarr` store (not necessarily the `.zarr` path itself—code uses `os.path.join(data_path_1, dataset_name_1)`).
- Set `--dataset_name_1` to the folder name (e.g. `EM_StructureMaps.ome.zarr`).
- Arrays are read as `store["0"]` with shape **`(C, H, W)`**.
- Channels:
  - **`--haadf_channel_1`**: HAADF/EM index in `C`
  - **`--target_channel_1`**: structure map index in `C`

Patch extraction size and stride:

- **`--patch_size`** (default 512)
- **`--overlap`**: fractional overlap between sliding windows

Training may filter patches using **`--target_presence_threshold`** and (when set) **`--keep_ratio`** (see `_load_single_dataset` in `data/omezarr_patch_dataset.py`). For testing, you can set **`--keep_ratio 0`** in `options/test_options.py` so more tiles are retained (depending on HAADF validity mask).

## Tile splits and structure presets

On a **fixed 132-tile** grid per EM dataset, positive train/test tiles depend on **which biological structure / channel preset** you use.

- Presets are defined at the **bottom** of **`data/omezarr_patch_dataset.py`** in `_STRUCTURE_TILE_PRESETS` (keys: `nucleic_acids`, `zymogen`, `insulin`, `lysosome`, `glucagon`).
- Default preset name is **`_ACTIVE_STRUCTURE_PRESET`** in the same file.
- Override from the CLI when training:

  ```bash
  --structure_preset glucagon
  ```

Training writes a JSON split file:

```text
data/tile_split_indices_<preset>.json
```

It contains keys such as `train_positive_indices_1`, `test_positive_indices_1`, `train_negative_indices_1`, `test_negative_indices_1`, plus `splits_1`. **Point your evaluation script at the same JSON** that matches the model you trained.

> **Note:** `test.py` currently loads a JSON path set inside the script. Update that path so it matches `data/tile_split_indices_<your_preset>.json` under this project (or an absolute path you prefer).

## Training

`train.py` uses **`OmeZarrPatchDataset`** directly with a custom **`my_collate`** for batching `haadf` and `structuremap`.

Example (GPU 0):

```bash
python train.py --name my_exp --gpu_ids 0 ^
  --data_path_1 /path/to/zarr_parent ^
  --dataset_name_1 EM_StructureMaps.ome.zarr ^
  --haadf_channel_1 2 --target_channel_1 1 ^
  --structure_preset insulin ^
  --batch_pos 4 --batch_neg 4
```

Example (CPU):

```bash
python train.py --name my_exp --gpu_ids -1 --data_path_1 ... --dataset_name_1 ...
```

Where outputs go:

- **Checkpoints**: `./checkpoints/<name>/` (net weights, `opt.txt`, optional `iter.txt` for resume)
- **HTML snapshots** (unless `--no_html`): `./checkpoints/<name>/web/`
- **TensorBoard** (optional): `./checkpoints/<name>/logs/` with `--tf_log`

Resume:

```bash
python train.py --continue_train --name my_exp ...
```

### Useful training flags

| Area | File | Notes |
|------|------|--------|
| Data paths, channels, patch size | `options/base_options.py` | `--data_path_1`, `--dataset_name_1`, `--patch_size`, `--overlap`, … |
| Structure preset | `options/train_options.py` | `--structure_preset` |
| Intensity-aware crop on positives | `options/train_options.py` | `--intensity_threshold`, `--max_crop_attempts` |
| Schedule / optimizer | `options/train_options.py` | `--niter`, `--niter_decay`, `--lr`, … |
| Batch composition | `options/train_options.py` | `--batch_pos`, `--batch_neg` |

## Testing / evaluation

`test.py` loads **`TestOptions`**, builds the dataloader via **`CreateDataLoader` → `OmeZarrPatchDataset`**, runs the generator, saves all tiles to an HTML page, and for **positive test tiles** computes **PCC** (Pearson correlation on normalized patches) and writes CSV + `evaluation_metrics.txt`.

Example:

```bash
python test.py --name my_exp --phase test --which_epoch latest ^
  --gpu_ids 0 --results_dir ./results/ ^
  --data_path_1 /path/to/zarr_parent --dataset_name_1 EM_StructureMaps.ome.zarr
```

Adjust **`--how_many`**, **`--keep_ratio`**, and ensure **`test.py`**’s **`json_path`** matches the split JSON produced during training for the same preset.

Outputs:

- **All tiles (HTML)**: `./results/<name>/test_<epoch>/index.html`
- **Positive-tile crops + metrics**: under `./results/<name>/..._pos_only_metrics/` and metrics next to the web folder

## Project layout (short)

```text
train.py                 # Training loop (OmeZarrPatchDataset + collate)
test.py                  # Inference + PCC evaluation (edit JSON path inside)
data/omezarr_patch_dataset.py   # OME-Zarr patches, augmentations, tile presets + JSON export
data/custom_dataset_data_loader.py   # Loader used by test.py
models/                  # pix2pixHD variant (generator / discriminators / losses)
options/                 # base_options.py, train_options.py, test_options.py
checkpoints/, results/   # Created at runtime
```

## Citation

If you find this useful for your research, please use the following.:

```bibtex
@inproceedings{tianPRISM,
  title={ColorEM-assisted deep learning for label-free data-driven segmentation of cellular ultrastructure},
  author={Changmin Tian, B.H. Peter Duinkerken, Jacob P. Hoogenboom, George Azzopardi*, Ben N.G. Giepmans*, Ahmad M.J. Alsahaf*},
  journal={*******},
  year={****}
}
```

## Acknowledgments
<img width="1080" height="1080" alt="www alltoall net_video_project1080_vK2jemi0k1" src="https://github.com/user-attachments/assets/8e51b55a-1376-424d-a7c3-785d87b139ac" />


Code builds on **[pix2pixHD](https://github.com/NVIDIA/pix2pixHD)** (NVIDIA Research)
