"""
The implementation of the function is heavily borrowed from 
https://github.com/NVIDIA/FastPhotoStyle/blob/af0c8fecce58aa71f76488546231214f6684be02/photo_gif.py#L13
We thank NVIDIA for sharing this code.
"""
from __future__ import division
from PIL import Image
from torch import nn
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter


class GIFSmoothing(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, r, eps):
        super(GIFSmoothing, self).__init__()
        self.r = r
        self.eps = eps

    def process(self, initImg, contentImg):
        return self.process_opencv(initImg, contentImg)

    def process_opencv(self, initImg, contentImg):
        '''
        :param initImg: intermediate output. Either image path or PIL Image
        :param contentImg: content image output. Either path or PIL Image
        :return: stylized output image. PIL Image
        '''
        if type(initImg) == str:
            init_img = cv2.imread(initImg)  # cv2.imread load image with BGR channels
            init_img = init_img[2:-2, 2:-2, :]
        else:
            init_img = np.array(initImg)[:, :, ::-1].copy()  # convert RGB to BGR

        if type(contentImg) == str:
            cont_img = cv2.imread(contentImg)  #  cv2.imread load image with BGR channels
        else:
            cont_img = np.array(contentImg)[:, :, ::-1].copy()  # convert RGB to BGR

        output_img = guidedFilter(
            guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)  # convert back to RGB
        output_img = Image.fromarray(output_img)
        return output_img
