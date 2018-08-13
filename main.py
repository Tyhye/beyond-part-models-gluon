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
    --feature_channels=<int>    Feature channels of the network [default: 256]
    --partnum=<int>             The number of the pcb parts. [default: 6]
    --feature_weight_share      If the six partnum share weights.
    --base_not_pretrained       If the base network don't pretrained on ImageNet
    --pretrain_path=<str>       Path to pretrained model. 

Training Setting Options:
    --optim=<str>               Optimizer Type [default: sgd]
    --lr_policy=<str>           Learning rate policy [default: multistep]
    --milestones=<list>         Step milestone for multistep policy [default: [40,]]
    --gamma=<float>             Gamma for multistep policy [default: 0.1]
    
    --max_epochs=<int>          Max Train epochs [default: 60]
    --val_epochs=<int>          Val step stone [default: 5]
    --snap_epochs=<int>         Snap step stone [default: 5]
    --snap_dir=<str>            Model state dict file path [default: saved/]

Data Options:
    --resize_size=<tuple>       Image resize size tuple (width, height) [default: (128, 384)]
    --crop_size=<tuple>         Image crop size tuple (width, height) [default: (128, 384)]
    --batch_size=<int>          Batchsize [default: 32]
    --feature_norm              If the feature are normalized when testing.

Train Data Options:
    --train_list=<str>          Train files list txt [default: datas/Market1501/train.txt]
    --train_data_root=<str>     Train sketch images path prefix [default: datas/Market1501/]
    
Test Data Options:
    --query_list=<str>          Query files list txt [default: datas/Market1501/query.txt]
    --query_data_root=<str>     Query sketch images path prefix [default: datas/Market1501/]
    --gallery_list=<str>        Gallery files list txt [default: datas/Market1501/gallery.txt]
    --gallery_data_root=<str>   Gallery sketch images path prefix [default: datas/Market1501/]
    
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

    cfg.optim = args['--optim']
    if cfg.optim == 'sgd':
        cfg.momentum = float(args['--momentum'])
    cfg.lr_policy = args['--lr_policy']
    if cfg.lr_policy == "multistep" or cfg.lr_policy == "multifactor":
        cfg.milestones = eval(args['--milestones'])
        cfg.gamma = float(args['--gamma'])

    cfg.max_epochs = int(args['--max_epochs'])
    cfg.val_epochs = int(args['--val_epochs'])
    cfg.snap_epochs = int(args['--snap_epochs'])
    if cfg.snap_epochs % cfg.val_epochs != 0:
        raise "Because the saver should use the val result, so the snap_epochs must be times of val_epochs"
    cfg.snap_dir = args['--snap_dir']
    if not os.path.exists(cfg.snap_dir):
        os.makedirs(cfg.snap_dir)
    
    cfg.batch_size = int(args['--batch_size'])
    cfg.resize_size = eval(args['--resize_size'])
    cfg.crop_size = eval(args['--crop_size'])
    cfg.feature_norm = args['--feature_norm']

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
    else:
        cfg.rpp_train = cfg.base_train

    cfg.train_list=args['--train_list']
    cfg.train_data_root =args['--train_data_root']
    cfg.query_list=args['--query_list']
    cfg.query_data_root=args['--query_data_root']
    cfg.gallery_list=args['--gallery_list']
    cfg.gallery_data_root=args['--gallery_data_root']

    from experiment.train_pcbrpp import train_pcbrpp
    train_pcbrpp(cfg, logprint)


if __name__ == "__main__":
    main()