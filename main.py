#!/usr/bin/env python 
# -*- code:utf-8 -*- 
'''
 @Author: tyhye.wang 
 @Date: 2018-08-05 10:14:10 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-05 10:14:10 
'''

import docopt 
import time
import os
import logging


docstr = """Train <Beyond Part Models: Person Retrieval with Refined Part Pooling>.

Usage: 
    python main.py [options]
    python main.py --withpcb [options]
    python main.py --withpcb --withrpp [options]

General Options:
    -h, --help                  Print this message
    --device=<str>              Device for runnint the model [default: cuda:0]
    --log=<str>                 File path for saving log message. 

Network Options:
    --basenet_type=<str>        BaseNet type for Model [default: resnet50]
    --classes_num=<int>         Output classes number of the network [default: 751]
    --feature_channels=<int>    Feature channels of the network [default: 256]
    --partnum=<int>             The number of the pcb parts. [default: 6]
    --feature_weight_share      If the six partnum share weights.

Snap and Pretrain Options:
    --Snap=<str>                Model state dict file path [default: saved/]
    --basepretrained            If the base network pretrained on ImageNet [default: True]
    --pretrain_path=<str>       Path to pretrained model. 

Training Setting Options:
    --resize_size=<tuple>       Image resize size tuple (height, width) [default: (384, 128)]
    --crop_size=<tuple>         Image crop size tuple (height, width) [default: (384, 128)]
    --batchsize=<int>           Batchsize [default: 8]
    
    --Optim=<str>               Optimizer Type [default: SGD]
    --LRpolicy=<str>            Learning rate policy [default: multistep]
    --Stones=<str>              Step stone for multistep policy [default: [40,]]

    --max_epochs=<int>          Max Train epochs [default: 60]
    --log_epochs=<int>          Log step stone [default: 5]
    --snap_epochs=<int>         Snap step stone [default: 5]

Train Data Options:
    --trainList=<str>           Train files list txt [default: datas/Market1501/train.txt]
    --trainIMpath=<str>         Train sketch images path prefix [default: datas/img_gt/]
    
Test Data Options:
    --queryList=<str>           Query files list txt [default: datas/Market1501/query.txt]
    --queryIMpath=<str>         Query sketch images path prefix [default: datas/Market1501/]
    --galleryList=<str>         Gallery files list txt [default: datas/Market1501/gallery.txt]
    --galleryIMpath=<str>       Gallery sketch images path prefix [default: datas/Market1501/]
    
Learning Rate Options:
    --learning_rate=<float>     Learning rate for training process [default: 0.01]      
    --base_not_train            If don't train base network.
    --base_lr_scale=<float>     Learing rate scale rate for the base network [default: 0.1]
    --pcb_not_train             If the pcb module or the tail in `IDE` are not trained.
    --pcb_lr_scale=<float>      Learing rate scale rate for the pcb module or the tail. [default: 1.0]
    --rpp_not_train             If don't train the rpp module.
    --rpp_lr_scale=<float>      Learing rate scale rate for the rpp module. [default: 1.0]

"""

