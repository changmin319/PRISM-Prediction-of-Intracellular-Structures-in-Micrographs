#base_options.py
import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='glucagon_threshold_0316', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes       
        # self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')

        # for setting inputs
        # Dataset 1
        self.parser.add_argument('--data_path_1', type=str, default='D:/cp1_datasets',
                 help='Path to OME-Zarr dataset')
        self.parser.add_argument('--dataset_name_1', type=str, default='new_5_masks_haadf_18_312.ome.zarr',
                 help='Name of OME-Zarr file')
        self.parser.add_argument('--haadf_channel_1', type=int, default= 2,
                 help='HAADF channel index for dataset')
        self.parser.add_argument('--target_channel_1', type=int, default= 1,
                 help='Structure map channel index in OME-Zarr')
        self.parser.add_argument('--manual_train_indices_1', type=str, default='',
                 help='Manual train indices for dataset, e.g. "0,1,2"')
        self.parser.add_argument('--manual_test_indices_1', type=str, default='',
                 help='Manual test indices for dataset, e.g. "3,4,5"')

        # General settings
        # self.parser.add_argument('--combine_strategy', type=str, default='concat', choices=['concat', 'interleave'],
        #          help='Strategy for combining multiple datasets: concat (sequential), interleave (alternating)')
        self.parser.add_argument('--target_presence_threshold', type=float, default=0.0,
                 help='Minimum fraction of nonzero pixels in target patch to keep it')
        # self.parser.add_argument('--validate_dataset_compatibility', action='store_true', default=True,
        #          help='Validate that all datasets have compatible dimensions and channels')
        # self.parser.add_argument('--dataset_info', action='store_true', default=False,
        #          help='Print detailed information about all datasets being used')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')                
        # self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        


        # #
        # self.parser.add_argument('--offline_augment', action='store_true',
        #                          help='If specified, replicate each patch with all chosen angles/flips. Increases dataset size.')
        # self.parser.add_argument('--augment_angles', type=int, nargs='+',
        #                          default=[90, 180, 225, 270, 300],
        #                          help='List of angles for offline rotation duplication.')
        # self.parser.add_argument('--use_hflip', action='store_true',
        #                          help='If specified, do horizontal flip for duplication.')
        # self.parser.add_argument('--use_vflip', action='store_true',
        #                          help='If specified, do vertical flip for duplication.')
        # self.parser.add_argument('--use_diagonal_flip', action='store_true',
        #                          help='If specified, apply diagonal flipping (transpose) for offline augmentation.')

        # BaseOptions / TrainOptions / TestOptions
        # self.parser.add_argument('--visualize_grid', action='store_true',
        #                          help='Show patch grid once when the dataset is built')

        # New options for OME-Zarr and patch-based processing
        self.parser.add_argument('--patch_size', type=int, default=512,
                                 help='Size of the patches extracted from the data.')
        self.parser.add_argument('--overlap', type=float, default=0.0,
                                 help='Overlap ratio for the patches (e.g., 0.1 for 10%).')

        # Add missing resize_or_crop argument for compatibility with pix2pixHD_model
        # self.parser.add_argument('--resize_or_crop', type=str, default='none',
        #          help='[compat] Resize/crop mode for images. Default is none.')

        self.parser.set_defaults(no_vgg_loss=True)

        self.initialized = True


    def parse(self, save=True, args=None):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=args)
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args_dict = vars(self.opt)

        print('Options')
        for k, v in sorted(args_dict.items()):
            print('%s: %s' % (str(k), str(v)))
        print('End')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('Options\n')
                for k, v in sorted(args_dict.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('End\n')
        return self.opt