#!/usr/bin/env python 
# -*- code:utf-8 -*- 
'''
 @Author: tyhye.wang 
 @Date: 2018-08-05 17:12:29 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-08-05 17:12:29 

 I feel the native LR scheduler is not inconvenient, so I write a custom one.
'''

class MultiStepListScheduler(object):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        trainers (Optimizers): Wrapped trainer list.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(trainer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, trainers, milestones, gamma=0.1): #, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = -1
        if isinstance(trainers, (tuple, list)):
            self.trainers = trainers
        else:
            raise "trainers should be a list."

    def step(self):
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            for t in self.trainers:
                t.set_learning_rate(t.learning_rate * self.gamma)