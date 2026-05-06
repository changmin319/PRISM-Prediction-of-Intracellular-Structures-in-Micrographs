import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5


    vflip = False
    if opt.use_vertical_flip:
        vflip = (random.random() > 0.5)

    angle = 0
    if opt.use_rotation:
        angle_candidates = [45, 90, 135, 180, 225, 270]
        angle = random.choice(angle_candidates)

    params = {
        'crop_pos': (x, y),
        'flip': flip,
        'vflip': vflip,
        'angle': angle
    }
    return params

    # return {'crop_pos': (x, y), 'flip': flip}

class TransformTracker:
    def __init__(self):
        self.flip_count = 0
        self.total_count = 0

    def record(self, flip):
        self.total_count += 1
        if flip:
            self.flip_count += 1

transform_tracker = TransformTracker()

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    # 2) rotation
    if opt.isTrain and opt.use_rotation:
        angle = params['angle']  # get_params
        if angle != 0:
            transform_list.append(transforms.Lambda(lambda img: __rotate(img, angle)))

    # 3)
    if opt.isTrain:

        if not opt.no_flip:
            transform_list.append(transforms.Lambda(lambda img: __flip_horizontal(img, params['flip'])))

        if opt.use_vertical_flip:
            transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['vflip'])))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_tracker.record(params['flip'])
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))    #data augmentation

    print(f"Total samples loaded: {transform_tracker.total_count}")
    print(f"Samples flipped: {transform_tracker.flip_count}")

    transform_list += [transforms.ToTensor()]

    if normalize:
        if opt.input_nc == 1:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    # if normalize:
    #     transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
    #                                             (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

# def normalize():
#     return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __rotate(img, angle):
    return img.rotate(angle, expand=True)

def __flip_horizontal(img, do_flip):
    if do_flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __flip_vertical(img, do_vflip):
    if do_vflip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img