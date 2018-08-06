#!/usr/bin/env python 
# -*- code:utf-8 -*- 
'''
 @Author: tyhye.wang 
 @Date: 2018-08-05 10:14:10 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-05 10:14:10 
'''

import time
import os
import logging
from docopt import docopt
from easydict import EasyDict as edict
import re

docstr = """Train <Beyond Part Models: Person Retrieval with Refined Part Pooling>.

Usage: 
    main.py [options]
    main.py --withpcb [options]
    main.py --withpcb --withrpp [options]

General Options:
    -h, --help                  Print this message
    --logfile=<str>             File path for saving log message. 
    --device_type=<str>         Device Type for running the model [default: cpu]
    --device_id=<int>           Device ID for running the model [default: 0]
    
Network Options:
    --basenet_type=<str>        BaseNet type for Model [default: resnet50_v2]
    --classes_num=<int>         Output classes number of the network [default: 751]
    --laststride=<int>          The stride of the last module in the base network [default: 2]
    --feature_channels=<int>    Feature channels of the network [default: 512]
    --partnum=<int>             The number of the pcb parts. [default: 6]
    --feature_weight_share      If the six partnum share weights.
    --base_not_pretrained       If the base network don't pretrained on ImageNet
    --pretrain_path=<str>       Path to pretrained model. 

Training Setting Options:
    --Optim=<str>               Optimizer Type [default: sgd]
    --LRpolicy=<str>            Learning rate policy [default: multistep]
    --milestones=<list>         Step milestone for multistep policy [default: [40,]]
    --gamma=<float>             Gamma for multistep policy [default: 0.1]
    
    --max_epochs=<int>          Max Train epochs [default: 60]
    --val_epochs=<int>          Val step stone [default: 5]
    --snap_epochs=<int>         Snap step stone [default: 5]
    --Snap=<str>                Model state dict file path [default: saved/]

Data Options:
    --resize_size=<tuple>       Image resize size tuple (height, width) [default: (384, 128)]
    --crop_size=<tuple>         Image crop size tuple (height, width) [default: (384, 128)]
    --batchsize=<int>           Batchsize [default: 32]

Train Data Options:
    --trainList=<str>           Train files list txt [default: datas/Market1501/train.txt]
    --trainIMpath=<str>         Train sketch images path prefix [default: datas/Market1501/]
    
Test Data Options:
    --queryList=<str>           Query files list txt [default: datas/Market1501/query.txt]
    --queryIMpath=<str>         Query sketch images path prefix [default: datas/Market1501/]
    --galleryList=<str>         Gallery files list txt [default: datas/Market1501/gallery.txt]
    --galleryIMpath=<str>       Gallery sketch images path prefix [default: datas/Market1501/]
    
Learning Rate Options:
    --learning_rate=<float>     Learning rate for training process [default: 0.01]
    --weight_decay=<float>      Weight decay for training process [default: 0.0005]
    --momentum=<float>          Momentum for the SGD Optimizer [default: 0.9]

    --base_not_train            If don't train base network.
    --base_lr_scale=<float>     Learing rate scale rate for the base network [default: 0.1]
    
    --tail_not_train            If don't train tail module, when w/o pcb and w/o rpp.
    --tail_lr_scale=<float>     Learing rate scale rate for the tail module.
    
    --rpp_not_train             If don't train the rpp module.
    --rpp_lr_scale=<float>      Learing rate scale rate for the rpp module.

"""

def main():
    args = docopt(docstr, version="v0.1")
    
    # -------set logger --------------------------------------------------------
    log_level = logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(level=log_level)
    formatter = logging.Formatter(
        '%(asctime)s-%(name)s-%(levelname)s\t-%(message)s')
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.INFO)
    consolehandler.setFormatter(formatter)
    logger.addHandler(consolehandler)
    if args['--logfile'] is not None:
        filehandler = logging.FileHandler(args['--logfile'], mode='w')
        filehandler.setLevel(log_level)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logprint = logger.info
    logprint(args)
    # ----------------- process configure --------------------------------------
    cfg = edict()
    cfg.withpcb = args['--withpcb']
    cfg.withrpp = args['--withrpp']
    if cfg.withrpp and not cfg.withpcb:
        raise "If setting withrpp, must setting withpcb"

    cfg.device_type = args['--device_type']
    cfg.device_id = int(args['--device_id'])
    
    cfg.basenet = args['--basenet_type']
    cfg.classes_num = int(args['--classes_num'])
    cfg.laststride = int(args['--laststride'])
    cfg.feature_channels = int(args['--feature_channels'])
    if cfg.withpcb:    
        cfg.partnum = int(args['--partnum'])
        cfg.feature_weight_share = args['--feature_weight_share']
    else:
        cfg.partnum = None
        cfg.feature_weight_share = True
    cfg.base_pretrained =  (not args['--base_not_pretrained']) and (args['--pretrain_path'] is None)
    cfg.pretrain_path = args['--pretrain_path']

    cfg.optim = args['--Optim']
    if cfg.optim == 'sgd':
        cfg.momentum = float(args['--momentum'])
    cfg.lrpolicy = args['--LRpolicy']
    if cfg.lrpolicy == "multistep" or cfg.lrpolicy == "multifactor":
        cfg.milestones = eval(args['--milestones'])
        cfg.gamma = float(args['--gamma'])

    cfg.max_epochs = int(args['--max_epochs'])
    cfg.val_epochs = int(args['--val_epochs'])
    cfg.snap_epochs = int(args['--snap_epochs'])
    if cfg.snap_epochs % cfg.val_epochs != 0:
        raise "Because the saver should use the val result, so the snap_epochs must be times of val_epochs"
    cfg.snapdir = args['--Snap']
    if not os.path.exists(cfg.snapdir):
        os.makedirs(cfg.snapdir)
    
    cfg.batchsize = int(args['--batchsize'])
    cfg.resize_size = eval(args['--resize_size'])
    cfg.crop_size = eval(args['--crop_size'])

    cfg.learning_rate = float(args['--learning_rate'])
    cfg.weight_decay = float(args['--weight_decay'])
    
    cfg.base_train = not args['--base_not_train']
    if cfg.base_train:
        cfg.base_learning_rate = cfg.learning_rate
        if args['--base_lr_scale'] is not None:
            cfg.base_learning_rate *= float(args['--base_lr_scale'])
    
    cfg.tail_train = not args['--tail_not_train']
    if cfg.tail_train:
        cfg.tail_learning_rate = cfg.learning_rate
        if args['--tail_lr_scale'] is not None:
            cfg.tail_learning_rate *= float(args['--tail_lr_scale'])
    
    if cfg.withrpp:
        cfg.rpp_train = not args['--rpp_not_train']
        if cfg.rpp_train:
            cfg.rpp_learning_rate = cfg.learning_rate
            if args['--rpp_lr_scale'] is not None:
                cfg.rpp_learning_rate *= float(args['--pcb_lr_scale'])

    cfg.trainList=args['--trainList']
    cfg.trainIMpath=args['--trainIMpath']
    cfg.queryList=args['--queryList']
    cfg.queryIMpath=args['--queryIMpath']
    cfg.galleryList=args['--galleryList']
    cfg.galleryIMpath=args['--galleryIMpath']

    from experiment.train_pcbrpp import train_pcbrpp
    train_pcbrpp(cfg, logprint)


if __name__ == "__main__":
    main()