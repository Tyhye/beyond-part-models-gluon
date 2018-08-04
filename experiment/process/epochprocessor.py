#!/usr/bin/env python 
# -*- code:utf-8 -*- 
'''
 @Author: tyhye.wang 
 @Date: 2018-08-05 00:54:30 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-05 00:54:30 

This file refers to the https://github.com/pytorch/tnt/blob/master/torchnet/engine/engine.py

'''


class EpochProcessor(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, process, iterator, maxepoch):
        state = {
            'process': process,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)
                
                loss, output = state["process"](state['sample'])
                state['loss'] = loss
                state['output'] = output
                self.hook('on_forward', state)
                state['output'] = None
                state['loss'] = None

                state['t'] += 1
                self.hook('on_end_iter', state)
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)

    def test(self, process, iterator):
        state = {
            'process': process,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            loss, output = state["process"](state['sample'])
            state['loss'] = loss
            state['output'] = output
            self.hook('on_forward', state)
            state['output'] = None
            state['loss'] = None

            state['t'] += 1
        self.hook('on_end', state)