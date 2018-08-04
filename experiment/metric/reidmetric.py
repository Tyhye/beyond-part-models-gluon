#!/usr/bin/env python
# -*- code:utf-8 -*-
'''
 @Author: tyhye.wang 
 @Date: 2018-06-16 08:05:43 
 @Last Modified by:   tyhye.wang 
 @Last Modified time: 2018-06-16 08:05:43 
 
 One metric object extend from the Metric.
 This metric is designed for person re-id retrival
'''

from mxnet.metric import EvalMetric
from mxnet.metric import check_label_shapes

import numpy as np
from tqdm import tqdm


class ReID_Metric(EvalMetric):
    """The Metric for Re-ID evaluation.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """

    def __init__(self, isnorm=True, name='CMC&mAP',
                 output_names=None, label_names=None):
        self.isnorm = isnorm
        self.query_features = []
        self.query_labels = []
        self.query_cams = []
        self.gallery_features = []
        self.gallery_labels = []
        self.gallery_cams = []
        # self._kwargs = kwargs
        super(ReID_Metric, self).__init__(name=name,
                                          output_names=output_names,
                                          label_names=label_names)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.query_features.clear()
        self.query_labels.clear()
        self.query_cams.clear()
        self.gallery_features.clear()
        self.gallery_labels.clear()
        self.gallery_cams.clear()
        self.reidmetric = None

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, labels, features, cams, type='query'):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> NDArray
            name to array mapping for labels.

        preds : OrderedDict of str -> NDArray
            name to array mapping of predicted outputs.
        """
        pass
        # if self.output_names is not None:
        #     pred = [pred[name] for name in self.output_names]
        # else:
        #     pred = list(pred.values())

        # if self.label_names is not None:
        #     label = [label[name] for name in self.label_names]
        # else:
        #     label = list(label.values())

        # self.update(label, pred)

    def update(self, features, cams, labels, feature_type='query'):
        """Updates the features of the `type`.

        Parameters
        ----------
        features : list of `NDArray`

        cams: list of `NDArray`
            The camids of the data.

        labels : list of `NDArray`
            The labels of the data.

        feature_type: str
            `query` or `gallery`
        """
        self.reidmetric = None
        if feature_type == 'query':
            features_list = self.query_features
            cams_list = self.query_cams
            labels_list = self.query_labels
        else:
            features_list = self.gallery_features
            cams_list = self.gallery_cams
            labels_list = self.gallery_labels
        features = features.asnumpy()
        if self.isnorm:
            fnorm = np.sqrt(np.sum(features ** 2, axis=1, keepdims=True))
            features = features / fnorm
        features_list.append(features)
        cams_list.append(cams.asnumpy())
        labels_list.append(labels.asnumpy())

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        reid-metrics : CMC list & mAP
           Value of the evaluations.
        """
        if self.reidmetric is not None:
            return (self.name, self.reidmetric)
        else:
            query_features = np.concatenate(self.query_features, axis=0)
            query_cams = np.concatenate(self.query_cams, axis=0)
            query_labels = np.concatenate(self.query_labels, axis=0)
            gallery_features = np.concatenate(self.gallery_features, axis=0)
            gallery_cams = np.concatenate(self.gallery_cams, axis=0)
            gallery_labels = np.concatenate(self.gallery_labels, axis=0)
            query_size = len(query_labels)

            print("Now is Calculating CMC & mAP: ")
            CMC = np.zeros(len(gallery_labels)).astype('int32')
            ap = 0.0
            # distance = np.matmul(query_features, gallery_features.T)
            for i in tqdm(range(query_size), ncols=80):
                ap_tmp, CMC_tmp = evaluate_oneitem(query_features[i],
                                                   query_labels[i],
                                                   query_cams[i],
                                                   gallery_features,
                                                   gallery_labels,
                                                   gallery_cams)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                ap += ap_tmp
            CMC = CMC.astype('float')
            CMC = CMC/query_size  # average CMC
            mAP = ap/query_size
            self.reidmetric = (CMC, mAP)
            return (self.name, self.reidmetric)

# Evaluate function


def evaluate_oneitem(qf, ql, qc, gf, gl, gc):
    query = qf
    score = np.dot(gf, query)
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    CMC_tmp = compute_mAP_(index, good_index, junk_index)
    return CMC_tmp

# compute map


def compute_mAP_(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall*(old_precision + precision)/2
    return ap, cmc
