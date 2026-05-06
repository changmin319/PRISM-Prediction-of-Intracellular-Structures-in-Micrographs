import os
import numpy as np
import torch
import sys
from collections import OrderedDict
import json

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import cv2
from scipy.stats import pearsonr

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

opt.manual_train_indices_1 = []
opt.manual_test_indices_1 = []
opt.manual_train_indices = []
opt.manual_test_indices = []

json_path = os.path.join(PROJECT_ROOT, 'PRISM',
                         'tile_split_indices_insulin.json')

try:
    with open(json_path, 'r') as f:
        split_info = json.load(f)

    test_positive_indices = [int(i) for i in split_info["test_positive_indices_1"]]
    test_negative_indices = [int(i) for i in split_info["test_negative_indices_1"]]

    opt.manual_test_indices_1 = test_positive_indices + test_negative_indices

    positive_test_set = set(test_positive_indices)

    print(f"Loaded indices from {json_path}")
    print(f"Total positive test tiles: {len(test_positive_indices)}")
    print(f"Total negative test tiles: {len(test_negative_indices)}")
    print(f"Total test tiles to load: {len(opt.manual_test_indices_1)}")

except FileNotFoundError:
    print(f"ERROR: JSON split file not found at {json_path}")
    print("Please run train.py first to generate the split file.")
    sys.exit(1)
except KeyError:
    print(f"ERROR: JSON file at {json_path} is missing required keys (e.g., 'test_positive_indices_1')")
    sys.exit(1)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

output_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.which_epoch}_pos_only_metrics")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving positive-only metric images to: {output_dir}")

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
print(f"Saving all test images to: {web_dir}")

if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

pccs = []
detailed_results = []


def compute_pcc(real_np, pred_np):
    real_f = real_np.astype(np.float32).flatten() / 255.0
    pred_f = pred_np.astype(np.float32).flatten() / 255.0
    pcc_val, _ = pearsonr(real_f, pred_f)
    return pcc_val


print("\nStarting evaluation loop...")
positive_tiles_processed = 0
negative_tiles_skipped = 0

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    try:
        orig_idx = data['orig_idx'].item()
    except KeyError:
        print("ERROR: 'orig_idx' not in data batch. Dataset not returning 'orig_idx'.")
        print("Please ensure OmeZarrPatchDataset returns 'orig_idx' during testing.")
        break
    except Exception as e:
        print(f"Error accessing 'orig_idx': {e}")
        print("Data keys are:", data.keys())
        print("Value of data['orig_idx']:", data.get('orig_idx'))
        break

    generated = model.inference(data['haadf'], None, data['structuremap'])

    input_label_np = util.tensor2im(data['haadf'][0], normalize=True)
    real_np = util.tensor2im(data['structuremap'][0], normalize=True)
    pred_np = util.tensor2im(generated.data[0], normalize=True)

    img_path_name = f"tile{orig_idx:03d}"
    visuals = OrderedDict([
        ('input_haadf', input_label_np),
        ('real_structuremap', real_np),
        ('pred_structuremap', pred_np)
    ])
    visualizer.save_images(webpage, visuals, [img_path_name])

    if orig_idx not in positive_test_set:
        negative_tiles_skipped += 1
        continue

    positive_tiles_processed += 1
    print(f"Processing positive tile {positive_tiles_processed}/{len(test_positive_indices)} (orig_idx: {orig_idx})...")

    tile_id = f"tile{orig_idx:03d}"
    cv2.imwrite(os.path.join(output_dir, f"{tile_id}_haadf.png"), input_label_np)
    cv2.imwrite(os.path.join(output_dir, f"{tile_id}_real.png"), real_np)
    cv2.imwrite(os.path.join(output_dir, f"{tile_id}_pred.png"), pred_np)

    pcc_val = compute_pcc(real_np, pred_np)
    detailed_results.append({'Tile_ID': orig_idx, 'PCC': pcc_val})
    pccs.append(pcc_val)

webpage.save()
print(f"\nSaved all test images to HTML page: {web_dir}/index.html")

metrics_file_path = os.path.join(web_dir, 'evaluation_metrics.txt')

summary_lines = []
summary_lines.append("Evaluation summary (positive tiles only, PCC)")

if positive_tiles_processed > 0:
    summary_lines.append(f"Total positive tiles processed: {positive_tiles_processed}")
    summary_lines.append(f"Total negative tiles skipped: {negative_tiles_skipped}")
    summary_lines.append("")
    summary_lines.append(f"Average PCC: {np.mean(pccs):.4f}")
else:
    summary_lines.append("No positive tiles were processed. Check your indices and data.")

summary_content = "\n".join(summary_lines)

print("\n" + summary_content)

with open(metrics_file_path, 'w') as f:
    f.write(summary_content)
print(f"\nMetrics saved to: {metrics_file_path}")

if detailed_results:
    df_results = pd.DataFrame(detailed_results)
    df_results['PCC'] = df_results['PCC'].round(3)
    csv_filename = f"{opt.phase}_{opt.which_epoch}_individual_metrics.csv"
    csv_path = os.path.join(web_dir, csv_filename)
    df_results.to_csv(csv_path, index=False, float_format='%.3f')
    print(f"\nPer-tile PCC saved to: {csv_path}")

print("\nTesting complete.")
