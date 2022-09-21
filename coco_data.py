# -*- coding: utf-8 -*-
# Author    : Yufeng Shi

import h5py
import numpy as np

def range_data(data):
    data = np.array(data).astype(np.int) - 1
    num_data = data.shape[0]
    return np.reshape(data, (num_data,))

class mscoco():
    def __init__(self):
        DATA_PATH = '/data/upload/shiyufeng/MS_COCO/COCO.mat'
        TEXT_PATH = '/data/upload/shiyufeng/MS_COCO/annotations/COCO_BoW.npy'

        data = h5py.File(DATA_PATH) #IAll 原始图像  LAll 标签 XAll cnn-f提的4096特征  param  数据集划分
        LAll_tmp = data['LAll']
        param_tmp = data['param']

        param = {}
        param['indexQuery'] = range_data(param_tmp['indexQuery'])
        param['indexRetrieval'] = range_data(param_tmp['indexDatabase'])

        self.LAll = np.squeeze(np.transpose(np.array(LAll_tmp), (1, 0)))
        self.TAll = np.squeeze(np.load(TEXT_PATH)) #自己提的2000维 BoW text特征
        self.param = param

        data.close()

    def get_data(self, use_way):
        if use_way == 'query':
            return self.param['indexQuery'], self.TAll[self.param['indexQuery'],:], self.LAll[self.param['indexQuery'],:]
        if use_way == 'retrieval':
            return self.param['indexRetrieval'], self.TAll[self.param['indexRetrieval'],:], self.LAll[self.param['indexRetrieval'],:]