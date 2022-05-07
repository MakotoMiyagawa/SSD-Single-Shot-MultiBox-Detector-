# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:12:34 2022

@author: makoto.miyagawa
"""

"""
1. データの拡張処理を行うクラス
"""
import cv2
import numpy as np
from numpy import random

class Compose(object):
    def __init__(self, transforms):
        """
        Args:
            transforms (list[Transform]): 変換処理のリスト
        Example:
            >>> augmentations.Compose([
                 transforms.CenterCrop(10),
                 transforms.ToTensor(),
                 ]) 

        """
        self.transforms = transforms
    
    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

"""
2. ピクセルデータのint型をfloat32に変換するクラス
"""
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

"""
3.アノテーションデータの正規化を元の状態に戻すクラス
"""
class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        
        return image, boxes, labels

'''
4. 輝度（明るさ）をランダムに変化させるクラス
'''
class RandomBrightness(object):
    def __init__(Self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = tandom.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels




























