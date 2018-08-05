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

from myresnet import resnet18_v1, resnet34_v1, resnet50_v1, resnet101_v1, resnet152_v1
from myresnet import resnet18_v2, resnet34_v2, resnet50_v2, resnet101_v2, resnet152_v2


class PCBRPPNet(HybridBlock):
    """
    PCBRPPNet is the model of `Beyond Part Model`. 
    You could configure model `with` or `without` the modules (part convolutional block) and (reﬁned part pooling) 
    by `withpcb` and `withrpp`

    Params:
        ...
    """

    def __init__(self, basenetwork='resnet50_v2', pretrained="True",
                 feature_channels=512, classes=751,
                 withpcb='True', partnum=6, feature_weight_share=False,
                 withrpp='True', **kwargs):

        super(PCBRPPNet, self).__init__(**kwargs)
        basenetwork = eval(basenetwork)
        self.withpcb = withpcb
        self.withrpp = withrpp
        if self.withrpp and not self.withpcb:
            raise "If withrpp is True, with pcb must be True."
        self.feature_weight_share = feature_weight_share
        self.part_num = partnum

        self.conv = basenetwork(pretrained=pretrained, ctx=cpu())
        self.pool = nn.GlobalAvgPool2D()

        if not self.withpcb or self.feature_weight_share:
            self.feature = nn.Dense(feature_channels, activation=None,
                                    use_bias=False, flatten=True)
            self.feature_ = nn.HybridSequential(prefix='')
            with self.feature_.name_scope():
                self.feature_.add(nn.BatchNorm())
                self.feature_.add(nn.Activation('relu'))
            self.feature_.hybridize()
            self.classifier = nn.Dense(classes)
            self.feature.collect_params().initialize(init=init.Xavier(), ctx=cpu())
            self.feature_.initialize(init=init.Zero(), ctx=cpu())
            self.classifier.collect_params().initialize(init=init.Normal(0.001), ctx=cpu())
        else:
            for pn in range(self.part_num):
                tmp_feature = nn.Dense(feature_channels, activation=None,
                                       use_bias=False, flatten=True)
                tmp_feature_ = nn.HybridSequential(prefix='')
                with tmp_feature_.name_scope():
                    tmp_feature_.add(nn.BatchNorm(center=False, scale=False))
                    tmp_feature_.add(nn.Activation('relu'))
                tmp_feature_.hybridize()
                tmp_classifier = nn.Dense(classes)

                tmp_feature.collect_params().initialize(init=init.Xavier(), ctx=cpu())
                tmp_feature_.collect_params().initialize(init=init.Zero(), ctx=cpu())
                tmp_classifier.collect_params().initialize(init=init.Normal(0.001), ctx=cpu())
                setattr(self, 'feature%d' % (pn+1), tmp_feature)
                setattr(self, 'feature%d_' % (pn+1), tmp_feature_)
                setattr(self, 'classifier%d' % (pn+1), tmp_classifier)

        if self.withrpp:
            self.rppscore = nn.Conv2D(
                self.partnum, kernel_size=1, use_bias=False)
            self.rppclassifier.collect_params().initialize(
                init=init.Normal(0.001), ctx=cpu())

    def hybrid_forward(self, F, x):
        x = self.conv(x)

        # ======================================================================
        # without pcb
        # ======================================================================
        if not self.withpcb:
            x = self.pool(x)
            fea = self.feature(x)
            x = self.feature_(fea)
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
            xs = [self.pool(x) for x in xs]
        else:
            xs = x.aplit(num_outputs=self.partnum, axis=1)

        # feature weight share or not
        if self.feature_weight_share:
            feas = [self.feature(x) for x in xs]
            xs = [self.feature_(fea) for fea in feas]
            IDs = [self.classifier(x) for x in xs]
        else:
            feas = [getattr(self, 'feature%d' % (pn+1))(x)
                    for (x, pn) in zip(xs, range(self.part_num))]
            xs = [getattr(self, 'feature%d_' % (pn+1))(fea)
                  for (fea, pn) in zip(feas, range(self.part_num))]
            IDs = [getattr(self, 'classifier%d' % (pn+1))(x)
                   for (x, pn) in zip(xs, range(self.part_num))]
        return IDs, feas
