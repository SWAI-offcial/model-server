#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: crnn_predict.py
@time: 2019/4/1 15:43
@desc:
'''
# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:21
# @Author  : zhoujun
from . import keys
import mxnet as mx
from mxnet import nd
from .crnn import CRNN

def try_gpu(gpu):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu)
        _ = nd.array([0], ctx=ctx)
    except BaseException as e:
        print(e)
        ctx = mx.cpu()
    return ctx

def decode(preds, alphabet, raw=False):
    results = []
    alphabet_size = len(alphabet)
    for word in preds:
        if raw:
            results.append(''.join([alphabet[int(i)] for i in word]))
        else:
            result = []
            for i, index in enumerate(word):
                if i < len(word) - 1 and word[i] == word[i + 1] and word[-1] != -1:  # Hack to decode label as well
                    continue
                if index == -1 or index == alphabet_size - 1:
                    continue
                else:
                    result.append(alphabet[int(index)])
            results.append(''.join(result))
    return results

class GluonNet_v1:
    def __init__(self, model_path,alphabet=keys.txt_alphabet, img_shape=(320, 32), img_channel=3,gpu_id=None):
        """
        初始化gluon模型
        :param model_path: 模型地址
        :param alphabet: 字母表
        :param img_shape: 图像的尺寸(w,h)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        """
        self.gpu_id = gpu_id
        self.img_w = img_shape[0]
        self.img_h = img_shape[1]
        self.img_channel = img_channel
        self.alphabet = alphabet
        self.ctx = try_gpu(gpu_id)
        print(self.ctx)
        self.net = CRNN(len(alphabet), hidden_size=512)
        self.net.load_parameters(model_path, self.ctx)
        self.net.hybridize()

    def predict(self, img):
        """
        对传入的图像进行预测，支持图像地址和numpy数组
        :param img_path: 图像地址
        :return:
        """
        img = nd.array(img)  # 深复制
        # img1 = transforms.ToTensor()(img)

        img1 = img.as_in_context(self.ctx)
        preds = self.net(img1)

        preds = preds.softmax().topk(axis=2).asnumpy()
        result = decode(preds, self.alphabet)
        return result, img


if __name__ == '__main__':
    import time
    from mxnet import gluon
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np

    font = FontProperties(fname=r"simsun.ttc", size=14)

    model_path = './output/test/99_0.9212_0.9890.params'
    alphabet = keys.txt_alphabet
    print(len(alphabet))
    gluon_net = GluonNet_v1(model_path=model_path, alphabet=alphabet, img_shape=(320, 32), img_channel=3)
    start = time.time()
    img = np.load('../images.npy')
    t = img[:,:,:,0]
    img[:, :, :, 0] = img[:,:,:,2]
    img[:, :, :, 2] = t
    img = np.transpose(img,(0,3,1,2)) / 255
    result, img = gluon_net.predict(img)
    print(time.time() - start)

