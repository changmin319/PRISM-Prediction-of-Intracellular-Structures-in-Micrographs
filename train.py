# train.py
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import math

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torchvision.utils as vutils

import random
import json
import zarr

from data.omezarr_patch_dataset import OmeZarrPatchDataset

torch.cuda.empty_cache()

# def lcm(a, b):
#     return abs(a * b) // math.gcd(a, b) if a and b else 0

def my_collate(batch):
    out = {}
    for k in batch[0]:
        if k in ('haadf', 'structuremap'):
            out[k] = torch.stack([b[k] for b in batch], 0)
        else:
            out[k] = [b[k] for b in batch]
    return out

def main():
    opt = TrainOptions().parse()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            # epoch_iter
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    opt.batchSize = opt.batch_pos + opt.batch_neg
    opt.offline_augment = False

    dataset = OmeZarrPatchDataset()
    dataset.initialize(opt)

    batch_size = opt.batch_pos + opt.batch_neg

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=opt.nThreads,
        collate_fn=my_collate,
        drop_last=False,
    )
    
    dataset_size = len(data_loader)
    print('#training batches = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    if opt.fp16:
        from apex import amp
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D


    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    # Training loop
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        
        for i, data in enumerate(data_loader):

            if epoch == start_epoch and i < epoch_iter:
                continue

            iter_start_time = time.time()

            total_steps += 1
            current_iter = i + 1

            save_fake = (total_steps % opt.display_freq == 0)

            # Forward pass
            losses, generated = model(Variable(data['haadf']), None,
                                      Variable(data['structuremap']), None, infer=save_fake)

            # Compute losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            # Backward pass and optimization
            optimizer_G.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_D.backward()
            optimizer_D.step()

            # Print errors
            if total_steps % opt.print_freq == 0:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, current_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            # Save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('Saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, i + 1), delimiter=',', fmt='%d')


            # Display output images
            if save_fake:
                visuals = OrderedDict([
                    ('input_label', util.tensor2im(data['haadf'][0], normalize=True)),
                    ('synthesized_image', util.tensor2im(generated.data[0], normalize=True)),
                    ('real_image', util.tensor2im(data['structuremap'][0], normalize=True))
                ])
                visualizer.display_current_results(visuals, epoch, total_steps)

            # Save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('Saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, i + 1), delimiter=',', fmt='%d')


        # End of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # Save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        # Update fixed params if needed
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()
        
        epoch_iter = 0

if __name__ == '__main__':
    main()