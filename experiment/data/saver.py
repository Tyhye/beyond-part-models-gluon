#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-06-29 19:08:30 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-06-29 19:08:30 
'''

import mxnet
import os


class Best_Evaluation_Saver(object):

    def __init__(self, save_dir, save_name=None, reverse=False):
        self.save_dir = save_dir
        self.save_name = save_name
        self.reverse = reverse
        self.best_evaluation = None

    def save(self, net, new_evaluation):
        if (self.best_evaluation is None) \
            or (not self.reverse and new_evaluation > self.best_evaluation) \
            or (self.reverse and new_evaluation < self.best_evaluation):
                if self.save_name is None:
                    fname = os.path.join(
                        self.save_dir, '%.4f.params' % (new_evaluation))
                else:
                    fname = os.path.join(self.save_dir, '%s_%.4f.params' % (
                        self.save_name, new_evaluation))
                net.save_params(fname)
                self.best_evaluation = new_evaluation