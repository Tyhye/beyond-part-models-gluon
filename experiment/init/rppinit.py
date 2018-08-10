#!/usr/bin/env python 
# -*- code:utf-8 -*- 
'''
 @Author: tyhye.wang 
 @Date: 2018-08-10 09:47:16 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-10 09:47:16 
'''
from mxnet.initializer import Initializer
import mxnet.random as random

class RPP_Init(Initializer):
    r"""Initializes weights with random values sampled from a normal distribution
    with a mean of `mean` and standard deviation of `std`.

    fullyconnected0_weight
    [[-0.3214761  -0.12660924  0.53789419]]
    """

    def __init__(self, mean=0.0, sigma=0.01):
        super(RPP_Init, self).__init__(mean=mean, sigma=sigma)
        self.mean = mean
        self.sigma = sigma

    def _init_weight(self, _, arr):
        tmparray = arr[0].copy()
        random.normal(self.mean, self.sigma, tmparray)
        arr[...] = tmparray.repeat(repeats=arr.size[0], axis=0)
