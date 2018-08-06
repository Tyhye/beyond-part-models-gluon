#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-06-12 16:10:34 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-06-12 16:10:34 
'''

from mxnet.gluon.data.vision import transforms

INTERPOLATION = 3  # NEAREST | BILINEAR | BICUBIC | ANTIALIAS
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ListTransformer(object):
    def __init__(self, istrain=False, datasetroot=None,
                 resize_size=None, crop_size=None,
                 interpolation=INTERPOLATION,
                 mean=MEAN, std=STD):
        self.datasetroot = datasetroot
        self.istrain = istrain
        transform_list = [ImageRead()]
        if resize_size is not None:
            transform_list.append(transforms.Resize(resize_size,
                                                    interpolation=interpolation))
        if crop_size is not None:
            if self.istrain:
                transform_list.append(RandomCrop(crop_size,
                                                 interpolation=interpolation))
            else:
                transform_list.append(transforms.CenterCrop(crop_size,
                                                            interpolation=interpolation))
        if self.istrain:
            transform_list.append(transforms.RandomFlipLeftRight())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))
        self.transformer = transforms.Compose(transform_list)

    def __call__(self, filepath):
        img_path, ID = filepath.strip().split(' ')
        if self.datasetroot is None:
            img = self.transformer(img_path)
        else:
            img = self.transformer('%s/%s' % (self.datasetroot, img_path))
        ID = int(ID)
        return img, ID


class Market1501_Transformer(object):
    def __init__(self, datasetroot=None, istrain=False,
                 resize_size=None, crop_size=None,
                 interpolation=INTERPOLATION, 
                 mean=MEAN, std=STD):
        self.datasetroot = datasetroot
        self.istrain = istrain
        transform_list = [ImageRead()]
        if resize_size is not None:
            transform_list.append(transforms.Resize(resize_size,
                                                    interpolation=interpolation))
        if crop_size is not None:
            if self.istrain:
                transform_list.append(RandomCrop(crop_size,
                                                 interpolation=interpolation))
            else:
                transform_list.append(transforms.CenterCrop(crop_size,
                                                            interpolation=interpolation))
        if self.istrain:
            transform_list.append(transforms.RandomFlipLeftRight())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))
        self.transformer = transforms.Compose(transform_list)

    def get_label_cam(self, file_path):
        filename = file_path.split('/')[-1]
        label = int(filename.split('_')[0])
        cam = int(filename.split('c')[-1].split('s')[0])
        return label, cam

    def __call__(self, filepath):
        img_path = filepath.strip()
        if self.datasetroot is None:
            img = self.transformer(img_path)
        else:
            img = self.transformer('%s/%s' % (self.datasetroot, img_path))
        label, cam = self.get_label_cam(img_path)
        return img, cam, label


from mxnet.gluon.block import Block
from mxnet import image, nd
from mxnet.base import numeric_types
from PIL import Image

class ImageRead(Block):
    """Read image by PIL.
    Parameters
    ----------

    Inputs:
        - filename: input image path
    Outputs:
        - **out**: output tensor with (H x W x C) shape of the filename
    """

    def __init__(self):
        super(ImageRead, self).__init__()
        pass

    def forward(self, filename):
        return nd.array(Image.open(filename).convert('RGB'))


class RandomCrop(Block):
    """Crop the input image in random place.
    Parameters
    ----------
    size : int or tuple of (W,H)
        Size of the final output.
    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
    Outputs:
        - **out**: output tensor with (H x W x C) shape.
    """

    def __init__(self, size, interpolation=2):
        super(RandomCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, interpolation)

    def forward(self, x):
        return image.random_crop(x, *self._args)[0]