#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-06-12 15:23:54 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-06-12 15:23:54 
'''
from mxnet.gluon.data import Dataset


class TextDataset(Dataset):
    """A dataset that define by the txt list.

    Every line of the txt is defined as one record(ie. 'filepath classid')
    the data and label is splited by transform

    Parameters
    ----------
    txtfilepath: path to the txt list file
    root: dataset root dir 
    transform: transform function which transform the record to what we want.
    """

    def __init__(self, txtfilepath, transform=None):
        self._transform = transform
        self._list_items(txtfilepath)

    def _list_items(self, txtfilepath):
        self.items = []
        with open(txtfilepath, 'r') as txtfile:
            txtlines = txtfile.readlines()
            for line in txtlines:
                self.items.append('%s' % line.strip())

    def __getitem__(self, idx):
        return self._transform(self.items[idx])

    def __len__(self):
        return len(self.items)
