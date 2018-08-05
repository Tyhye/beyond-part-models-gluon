#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-08-05 01:32:37 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-05 01:32:37 
'''

from __future__ import division

from tqdm import tqdm
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Trainer
from mxnet import ndarray
from mxnet import profiler
from mxnet.gluon import nn
from mxnet.gluon import ParameterDict
from mxnet.gluon.data import DataLoader
from mxnet import autograd
from mxnet.metric import Loss, Accuracy

from data.transform import ListTransformer, Market1501_Transformer
from data.textdataset import TextDataset
from model.pcbrpp import PCBRPPNet
from scheduler.listscheduler import MultiStepListScheduler
from process.epochprocessor import EpochProcessor

def train_pcbrpp(cfg, logprint=print):
    cfg.ctx = mx.Context(cfg.device_type, cfg.device_id)

    # ==========================================================================
    # define train dataset, query dataset and test dataset
    # ==========================================================================
    traintransformer = ListTransformer(datasetroot=cfg.trainList,
                                       resize_size=cfg.resize_size,
                                       crop_size=cfg.crop_size,
                                       istrain=True)
    querytransformer = Market1501_Transformer(datasetroot=cfg.queryList,
                                              resize_size=cfg.resize_size,
                                              crop_size=cfg.crop_size,
                                              istrain=False)
    gallerytransformer = Market1501_Transformer(datasetroot=cfg.queryList,
                                                resize_size=cfg.resize_size,
                                                crop_size=cfg.crop_size,
                                                istrain=False)
    traindataset = TextDataset(txtfilepath='../../Market-1501/dataset/train.txt',
                               transform=traintransformer),
    querydataset = TextDataset(txtfilepath='../../Market-1501/dataset/query.txt',
                               transform=querytransformer)
    gallerydataset = TextDataset(txtfilepath='../../Market-1501/dataset/gallery.txt',
                                 transform=gallerytransformer)
    train_iterator = DataLoader(traindataset, num_workers=1,
                                last_batch='discard', batch_size=cfg.batchsize,
                                shuffle=True)
    query_iterator = DataLoader(querydataset, num_workers=1,
                                last_batch='keep', batch_size=cfg.batchsize,
                                shuffle=True)
    gallery_iterator = DataLoader(gallerydataset, num_workers=1,
                                  last_batch='keep', batch_size=cfg.batchsize,
                                  shuffle=True)

    def test_iterator():
        for data in tqdm(query_iterator, ncols=80):
            if isinstance(data, (tuple, list)):
                data.append('query')
            else:
                data = (data, 'query')
            yield data
        for data in tqdm(gallery_iterator, ncols=80):
            if isinstance(data, (tuple, list)):
                data.append('gallery')
            else:
                data = (data, 'gallery')
            yield data
    # ==========================================================================

    # ==========================================================================
    # define model and trainer list, lr_scheduler
    # ==========================================================================
    Net = PCBRPPNet(basenetwork=cfg.basenet, pretrained=cfg.pretrained,
                    feature_channels=cfg.feature_channels,
                    classes=cfg.classes_num,
                    withpcb=cfg.withpcb, partnum=cfg.partnum,
                    feature_weight_share=cfg.feature_weight_share,
                    withrpp=cfg.withrpp)
    if cfg.pretrain_path is not None:
        Net.load_parameters(cfg.pretrain_path,
                            allow_missing=True, ignore_extra=True)
    Net.collect_params().reset_ctx(cfg.ctx)

    trainers = []
    if cfg.base_train:
        base_params = Net.conv.collect_params()
        base_optimizer_params = {'learning_rate': cfg.base_learning_rate,
                                 'weight_decay': cfg.weight_decay,
                                 'momentum': cfg.momentum,
                                 'multi_precision': True}
        basetrainer = Trainer(base_params, optimizer=cfg.optim,
                              optimizer_params=base_optimizer_params)
        trainers.append(basetrainer)
    if cfg.tail_train:
        tail_params = ParameterDict()
        if (not cfg.withpcb) or cfg.feature_weight_share:
            tail_params.update(Net.feature.collect_params())
            tail_params.update(Net.feature_.collect_params())
            tail_params.update(Net.classifier.collect_params())
        else:
            for pn in range(cfg.partnum):
                tail_params.update(
                    getattr(Net, 'feature%d' % (pn+1)).collect_params())
                tail_params.update(
                    getattr(Net, 'feature%d_' % (pn+1)).collect_params())
                tail_params.update(
                    getattr(Net, 'classifier%d' % (pn+1)).collect_params())
        tail_optimizer_params = {'learning_rate': cfg.tail_learning_rate,
                                 'weight_decay': cfg.weight_decay,
                                 'momentum': cfg.momentum,
                                 'multi_precision': True}
        tailtrainer = Trainer(tail_params, optimizer=cfg.optim,
                              optimizer_params=tail_optimizer_params)
        trainers.append(tailtrainer)
    if cfg.withrpp and cfg.rpp_train:
        rpp_params = Net.rppscore.collect_params()
        rpp_optimizer_params = {'learning_rate': cfg.rpp_learning_rate,
                                'weight_decay': cfg.weight_decay,
                                'momentum': cfg.momentum,
                                'multi_precision': True}
        rpptrainer = Trainer(rpp_params, optimizer=cfg.optim,
                             optimizer_params=rpp_optimizer_params)
        trainers.append(rpptrainer)
    if len(trainers) == 0:
        raise "There is no params for training."
    lr_scheduler = MultiStepListScheduler(trainers,
                                          milestones=cfg.milestones,
                                          gamma=cfg.gamma)
    # ==========================================================================

    # ==========================================================================
    # metric, loss, saver define
    # ==========================================================================
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_metric = Loss()
    if cfg.partnum is not None:
        train_accuracy_metrics = [Accuracy() for _ in range(part_num)]
    else:
        train_accracy_metric = Accuracy()
    reid_metric = ReID_Metric(isnorm=True)

    save_name = ""
    if not cfg.withpcb:
        save_name = "IDE"
    elif not cfg.withrpp:
        save_name = "NORPP_%dPart" % (cfg.partnum)
    else:
        if not cfg.tail_train and not cfg.base_train:
            save_name = "WITHRPP_%dPart" % (cfg.partnum)
    if cfg.withpcb and cfg.feature_weight_share:
        save_name += "_FEASHARE"
    net_saver = Best_Evaluation_Saver(save_dir=cfg.snapdir,
                                      save_name=save_name,
                                      reverse=False)
    # ==========================================================================
    def reset_metrics():
        loss_metric.reset()
        if cfg.partnum is not None:
            for train_accuracy_metric in train_accracy_metrics:
                train_accuracy_metric.reset()
        else:
            train_accracy_metric.reset()
        reid_metric.reset()

    def on_start(state):
        pass

    def on_start_epoch(state):
        pass
    
    def on_sample(state):
        pass

    def test_process(sample):
        pass
    
    def train_process(sample):
        pass
    
    def on_forward(state):
        pass

    def on_end_iter(state):
        pass
    
    def on_end_epoch(state):
        pass

    def on_end(state):
        pass

    processor = EpochProcessor()
    processor.hooks['on_start'] = on_start
    processor.hooks['on_start_epoch'] = on_start_epoch
    processor.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end_iter'] = on_end_epoch
    engine.hooks['on_end'] = on_end

    


