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

from .data.transform import ListTransformer, Market1501_Transformer
from .data.textdataset import TextDataset
from .data.saver import Best_Evaluation_Saver
from .model.pcbrpp import PCBRPPNet
from .scheduler.listscheduler import MultiStepListScheduler
from .process.epochprocessor import EpochProcessor
from .metric.reidmetric import ReID_Metric


def train_pcbrpp(cfg, logprint=print):
    cfg.ctx = mx.Context(cfg.device_type, cfg.device_id)

    # ==========================================================================
    # define train dataset, query dataset and test dataset
    # ==========================================================================
    traintransformer = ListTransformer(datasetroot=cfg.trainIMpath,
                                       resize_size=cfg.resize_size,
                                       crop_size=cfg.crop_size,
                                       istrain=True)
    querytransformer = Market1501_Transformer(datasetroot=cfg.queryIMpath,
                                              resize_size=cfg.resize_size,
                                              crop_size=cfg.crop_size,
                                              istrain=False)
    gallerytransformer = Market1501_Transformer(datasetroot=cfg.queryIMpath,
                                                resize_size=cfg.resize_size,
                                                crop_size=cfg.crop_size,
                                                istrain=False)
    traindataset = TextDataset(txtfilepath=cfg.trainList,
                               transform=traintransformer)
    querydataset = TextDataset(txtfilepath=cfg.queryList,
                               transform=querytransformer)
    gallerydataset = TextDataset(txtfilepath=cfg.galleryList,
                                 transform=gallerytransformer)
    train_iterator = DataLoader(traindataset, num_workers=1, shuffle=True,
                                last_batch='discard', batch_size=cfg.batchsize)
    query_iterator = DataLoader(querydataset, num_workers=1, shuffle=True,
                                last_batch='keep', batch_size=cfg.batchsize)
    gallery_iterator = DataLoader(gallerydataset, num_workers=1, shuffle=True,
                                  last_batch='keep', batch_size=cfg.batchsize)

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
    Net = PCBRPPNet(basenetwork=cfg.basenet, pretrained=cfg.base_pretrained,
                    feature_channels=cfg.feature_channels,
                    classes=cfg.classes_num, laststride=cfg.laststride,
                    withpcb=cfg.withpcb, partnum=cfg.partnum,
                    feature_weight_share=cfg.feature_weight_share,
                    withrpp=cfg.withrpp)
    if cfg.pretrain_path is not None:
        Net.load_params(cfg.pretrain_path, ctx=mx.cpu(),
                        allow_missing=True, ignore_extra=True)
    Net.collect_params().reset_ctx(cfg.ctx)

    trainers = []
    if cfg.base_train:
        base_params = Net.conv.collect_params()
        base_optimizer_params = {'learning_rate': cfg.base_learning_rate,
                                 'wd': cfg.weight_decay, 'momentum': cfg.momentum,
                                 'multi_precision': True}
        basetrainer = Trainer(base_params, optimizer=cfg.optim,
                              optimizer_params=base_optimizer_params)
        trainers.append(basetrainer)
    if cfg.tail_train:
        tail_params = ParameterDict()
        if (not cfg.withpcb) or cfg.feature_weight_share:
            tail_params.update(Net.feature.collect_params())
            # tail_params.update(Net.feature_.collect_params())
            tail_params.update(Net.classifier.collect_params())
        else:
            for pn in range(cfg.partnum):
                tail_params.update(
                    getattr(Net, 'feature%d' % (pn+1)).collect_params())
                # tail_params.update(
                #     getattr(Net, 'feature%d_' % (pn+1)).collect_params())
                tail_params.update(
                    getattr(Net, 'classifier%d' % (pn+1)).collect_params())
        tail_optimizer_params = {'learning_rate': cfg.tail_learning_rate,
                                 'wd': cfg.weight_decay, 'momentum': cfg.momentum,
                                 'multi_precision': True}
        tailtrainer = Trainer(tail_params, optimizer=cfg.optim,
                              optimizer_params=tail_optimizer_params)
        trainers.append(tailtrainer)
    if cfg.withrpp and cfg.rpp_train:
        rpp_params = Net.rppscore.collect_params()
        rpp_optimizer_params = {'learning_rate': cfg.rpp_learning_rate,
                                'wd': cfg.weight_decay, 'momentum': cfg.momentum,
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
        train_accuracy_metrics = [Accuracy() for _ in range(cfg.partnum)]
    else:
        train_accuracy_metric = Accuracy()
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
    logprint(Net)
    # ==========================================================================
    # process functions
    # ==========================================================================

    def reset_metrics():
        loss_metric.reset()
        if cfg.partnum is not None:
            for metric in train_accuracy_metrics:
                metric.reset()
        else:
            train_accuracy_metric.reset()
        reid_metric.reset()

    def on_start(state):
        pass
        if state['train']:
            state['store_iterator'] = state['iterator']

    def on_start_epoch(state):
        lr_scheduler.step()
        reset_metrics()
        if state['train']:
            state['iterator'] = tqdm(state['store_iterator'], ncols=80)

    def on_sample(state):
        pass

    def test_process(sample):
        img, cam, label, ds = sample
        img = img.as_in_context(cfg.ctx)
        ID1, Fea1 = Net(img)
        if cfg.partnum is not None:
            Fea1 = ndarray.concat(*Fea1, dim=-1)
        img = img.flip(axis=3)
        ID2, Fea2 = Net(img)
        if cfg.partnum is not None:
            Fea2 = ndarray.concat(*Fea2, dim=-1)
        return None, Fea1+Fea2

    def train_process(sample):
        data, label = sample
        data = data.as_in_context(cfg.ctx)
        label = label.as_in_context(cfg.ctx)
        with autograd.record():
            ID, Fea = Net(data)
            if isinstance(ID, list):
                losses = [softmax_cross_entropy(id_, label) for id_ in ID]
                loss = ndarray.stack(*losses, axis=0).mean(axis=0)
            else:
                loss = softmax_cross_entropy(ID, label)
        loss.backward()
        for trainer in trainers:
            trainer.step(data.shape[0])
        return loss, ID

    def on_forward(state):
        if state['train']:
            img, label = state['sample']
            loss_metric.update(None, state['loss'])
            if cfg.partnum is not None:
                for metric, id_ in zip(train_accuracy_metrics, state['output']):
                    metric.update(preds=id_, labels=label)
            else:
                train_accuracy_metric.update(
                    preds=state['output'], labels=label)
        else:
            img, cam, label, ds = state['sample']
            if cfg.feature_norm:
                fnorm = ndarray.power(state['output'], 2)
                fnorm = ndarray.sqrt(ndarray.sum(
                    fnorm, axis=-1, keepdims=True))
                state['output'] = state['output'] / fnorm
            reid_metric.update(state['output'], cam, label, ds)

    def on_end_iter(state):
        pass

    def on_end_epoch(state):
        if state['train']:
            logprint("[Epoch %d] train loss: %.6f" %
                     (state['epoch'], loss_metric.get()[1]))
            if cfg.partnum is not None:
                for idx, metric in enumerate(train_accuracy_metrics):
                    logprint("[Epoch %d] part No.%d train accuracy: %.2f%%" %
                             (state['epoch'], idx+1, metric.get()[1]*100))
            else:
                logprint("[Epoch %d] train accuracy: %.2f%%" %
                         (state['epoch'], train_accuracy_metric.get()[1]*100))
            if state['epoch'] % cfg.val_epochs == 0:
                reset_metrics()
                processor.test(test_process, test_iterator())
                CMC, mAP = reid_metric.get()[1]
                logprint("[Epoch %d] CMC1: %.2f%% CMC5: %.2f%% CMC10: %.2f%% CMC20: %.2f%% mAP: %.2f%%" %
                         (state['epoch'], CMC[0]*100, CMC[4]*100, CMC[9]*100, CMC[19]*100, mAP*100))
                if state['epoch'] % cfg.snap_epochs == 0:
                    net_saver.save(Net, CMC[0])

    def on_end(state):
        pass

    processor = EpochProcessor()
    processor.hooks['on_start'] = on_start
    processor.hooks['on_start_epoch'] = on_start_epoch
    processor.hooks['on_sample'] = on_sample
    processor.hooks['on_forward'] = on_forward
    processor.hooks['on_end_iter'] = on_end_iter
    processor.hooks['on_end_epoch'] = on_end_epoch
    processor.hooks['on_end'] = on_end

    processor.train(train_process, train_iterator, cfg.max_epochs)
