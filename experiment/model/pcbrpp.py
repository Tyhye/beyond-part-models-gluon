#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-06-12 20:21:51 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-06-12 20:21:51 
'''

import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import mxnet.initializer as init

from .myresnet import resnet18_v1, resnet34_v1, resnet50_v1, resnet101_v1, resnet152_v1
from .myresnet import resnet18_v2, resnet34_v2, resnet50_v2, resnet101_v2, resnet152_v2


class PCBRPPNet(HybridBlock):
    """
    PCBRPPNet is the model of `Beyond Part Model`. 
    You could configure model `with` or `without` the modules (part convolutional block) and (reﬁned part pooling) 
    by `withpcb` and `withrpp`

    Params:
        ...
    """

    def __init__(self, basenetwork='resnet50_v2', pretrained="True",
                 feature_channels=512, classes=751, laststride=2,
                 withpcb='True', partnum=6, feature_weight_share=False,
                 withrpp='True', **kwargs):

        super(PCBRPPNet, self).__init__(**kwargs)
        basenetwork = eval(basenetwork)
        self.withpcb = withpcb
        self.withrpp = withrpp
        if self.withrpp and not self.withpcb:
            raise "If withrpp is True, with pcb must be True."
        self.feature_weight_share = feature_weight_share
        self.partnum = partnum

        self.conv = basenetwork(pretrained=pretrained,
                                laststride=laststride, ctx=cpu())
        if not pretrained:
            self.conv.collect_params().initialize(init=init.Xavier(), ctx=cpu())

        self.pool = nn.GlobalAvgPool2D()
        self.dropout = nn.Dropout(rate=0.5)

        if not self.withpcb or self.feature_weight_share:
            self.feature = nn.HybridSequential(prefix='')
            with self.feature.name_scope():
                self.feature.add(nn.Dense(feature_channels, activation=None,
                                    use_bias=False, flatten=True))
                self.feature.add(nn.BatchNorm())
                self.feature.add(nn.LeakyReLU(alpha=0.1))
            self.feature.hybridize()
            self.classifier = nn.Dense(classes, use_bias=False)
            self.feature.collect_params().initialize(init=init.Xavier(), ctx=cpu())
            self.classifier.collect_params().initialize(init=init.Normal(0.001), ctx=cpu())
        else:
            for pn in range(self.partnum):
                tmp_feature = nn.Dense(feature_channels, activation=None,
                                       use_bias=False, flatten=True)
                tmp_classifier = nn.Dense(classes, use_bias=False)
                tmp_feature.collect_params().initialize(init=init.Xavier(), ctx=cpu())
                tmp_classifier.collect_params().initialize(init=init.Normal(0.001), ctx=cpu())
                setattr(self, 'feature%d' % (pn+1), tmp_feature)
                setattr(self, 'classifier%d' % (pn+1), tmp_classifier)

        if self.withrpp:
            # from ..init.rppinit import RPP_Init
            # rpp_init = RPP_Init(mean=0.0, sigma=0.001)
            self.rppscore = nn.Conv2D(
                self.partnum, kernel_size=1, use_bias=False)
            self.rppscore.collect_params().initialize(init=init.One(), ctx=cpu())
                

    def hybrid_forward(self, F, x):
        x = self.conv(x)

        # ======================================================================
        # without pcb
        # ======================================================================
        if not self.withpcb:
            x = self.pool(x)
            x = self.dropout(x)
            fea = self.feature(x)
            ID = self.classifier(x)
            return ID, fea

        # ======================================================================
        # with pcb
        # w or w/o rpp
        # ======================================================================
        if self.withrpp:
            rppscore = self.rppscore(x)
            rppscore = rppscore.softmax(axis=1)
            rppscores = rppscore.split(num_outputs=self.partnum, axis=1)
            xs = [score*x for score in rppscores]
        else:
            xs = x.split(num_outputs=self.partnum, axis=2)
        xs = [self.pool(x) for x in xs]
        xs = [self.dropout(x) for x in xs]

        # feature weight share or not
        if self.feature_weight_share:
            feas = [self.feature(x) for x in xs]
            IDs = [self.classifier(x) for x in feas]
        else:
            feas = [getattr(self, 'feature%d' % (pn+1))(x)
                    for (x, pn) in zip(xs, range(self.partnum))]
            IDs = [getattr(self, 'classifier%d' % (pn+1))(x)
                   for (x, pn) in zip(feas, range(self.partnum))]
        return IDs, feas
    
    def base_forward(self, x):
        x = self.conv(x)
        return x

    def split_forward(self, x):
        if self.withrpp:
            rppscore = self.rppscore(x)
            rppscore = rppscore.softmax(axis=1)
            rppscores = rppscore.split(num_outputs=self.partnum, axis=1)
            xs = [score*x for score in rppscores]
        else:
            xs = x.split(num_outputs=self.partnum, axis=2)
        return xs
    
    def tail_forward(self, xs):
        if not self.withpcb:
            x = self.pool(xs)
            x = self.dropout(x)
            fea = self.feature(x)
            ID = self.classifier(x)
            return ID, fea

        xs = [self.pool(x) for x in xs]
        xs = [self.dropout(x) for x in xs]
        if self.feature_weight_share:
            feas = [self.feature(x) for x in xs]
            IDs = [self.classifier(x) for x in feas]
        else:
            feas = [getattr(self, 'feature%d' % (pn+1))(x)
                    for (x, pn) in zip(xs, range(self.partnum))]
            IDs = [getattr(self, 'classifier%d' % (pn+1))(x)
                   for (x, pn) in zip(feas, range(self.partnum))]
        return IDs, feas
