# Source: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/bninception.py

from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

__all__ = ["BNInception", "bninception"]

pretrained_settings = {
    "bninception": {
        "imagenet": {
            # Was ported using python2 (may trigger warning)
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth",
            # 'url': 'http://yjxiong.me/others/bn_inception-9f5701afb96c8044.pth',
            "input_space": "BGR",
            "input_size": [3, 224, 224],
            "input_range": [0, 255],
            "mean": [104, 117, 128],
            "std": [1, 1, 1],
            "num_classes": 1000,
        }
    }
}


class BNInception(nn.Module):
    def __init__(self, num_classes=1000):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(
            64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv2_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_3a_pool_proj = nn.Conv2d(
            192, 32, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_3b_pool_proj = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(
            320, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(
            320, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(
            576, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4a_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4b_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(
            160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4c_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(
            128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(
            608, 160, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(
            160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4d_pool_proj = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(
            128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(
            608, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(
            192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(
            1056, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(
            192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(
            1056, 160, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(
            160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(
            224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_5a_pool_proj = nn.Conv2d(
            1056, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(
            1024, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(
            192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(
            1024, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(
            192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(
            224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d(
            (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True
        )
        self.inception_5b_pool_proj = nn.Conv2d(
            1024, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.last_linear = nn.Linear(1024, num_classes)

        self.feature_size = 1024

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(
            inception_3a_3x3_reduce_out
        )
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(
            inception_3a_3x3_reduce_bn_out
        )
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(
            pool2_3x3_s2_out
        )
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out
        )
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out
        )
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(
            inception_3a_relu_double_3x3_reduce_out
        )
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(
            inception_3a_double_3x3_1_out
        )
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(
            inception_3a_double_3x3_1_bn_out
        )
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(
            inception_3a_relu_double_3x3_1_out
        )
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(
            inception_3a_double_3x3_2_out
        )
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(
            inception_3a_double_3x3_2_bn_out
        )
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(
            inception_3a_pool_proj_out
        )
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(
            inception_3a_pool_proj_bn_out
        )
        inception_3a_output_out = torch.cat(
            [
                inception_3a_relu_1x1_out,
                inception_3a_relu_3x3_out,
                inception_3a_relu_double_3x3_2_out,
                inception_3a_relu_pool_proj_out,
            ],
            1,
        )
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(
            inception_3a_output_out
        )
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(
            inception_3b_3x3_reduce_out
        )
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(
            inception_3b_3x3_reduce_bn_out
        )
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(
            inception_3a_output_out
        )
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out
        )
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out
        )
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(
            inception_3b_relu_double_3x3_reduce_out
        )
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(
            inception_3b_double_3x3_1_out
        )
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(
            inception_3b_double_3x3_1_bn_out
        )
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(
            inception_3b_relu_double_3x3_1_out
        )
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(
            inception_3b_double_3x3_2_out
        )
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(
            inception_3b_double_3x3_2_bn_out
        )
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(
            inception_3b_pool_proj_out
        )
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(
            inception_3b_pool_proj_bn_out
        )
        inception_3b_output_out = torch.cat(
            [
                inception_3b_relu_1x1_out,
                inception_3b_relu_3x3_out,
                inception_3b_relu_double_3x3_2_out,
                inception_3b_relu_pool_proj_out,
            ],
            1,
        )
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(
            inception_3b_output_out
        )
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(
            inception_3c_3x3_reduce_out
        )
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(
            inception_3c_3x3_reduce_bn_out
        )
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(
            inception_3b_output_out
        )
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out
        )
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out
        )
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(
            inception_3c_relu_double_3x3_reduce_out
        )
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(
            inception_3c_double_3x3_1_out
        )
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(
            inception_3c_double_3x3_1_bn_out
        )
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(
            inception_3c_relu_double_3x3_1_out
        )
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(
            inception_3c_double_3x3_2_out
        )
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(
            inception_3c_double_3x3_2_bn_out
        )
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat(
            [
                inception_3c_relu_3x3_out,
                inception_3c_relu_double_3x3_2_out,
                inception_3c_pool_out,
            ],
            1,
        )
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(
            inception_3c_output_out
        )
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(
            inception_4a_3x3_reduce_out
        )
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(
            inception_4a_3x3_reduce_bn_out
        )
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(
            inception_3c_output_out
        )
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out
        )
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(
            inception_4a_double_3x3_reduce_bn_out
        )
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(
            inception_4a_relu_double_3x3_reduce_out
        )
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(
            inception_4a_double_3x3_1_out
        )
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(
            inception_4a_double_3x3_1_bn_out
        )
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(
            inception_4a_relu_double_3x3_1_out
        )
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(
            inception_4a_double_3x3_2_out
        )
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(
            inception_4a_double_3x3_2_bn_out
        )
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(
            inception_4a_pool_proj_out
        )
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(
            inception_4a_pool_proj_bn_out
        )
        inception_4a_output_out = torch.cat(
            [
                inception_4a_relu_1x1_out,
                inception_4a_relu_3x3_out,
                inception_4a_relu_double_3x3_2_out,
                inception_4a_relu_pool_proj_out,
            ],
            1,
        )
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(
            inception_4a_output_out
        )
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(
            inception_4b_3x3_reduce_out
        )
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(
            inception_4b_3x3_reduce_bn_out
        )
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(
            inception_4a_output_out
        )
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out
        )
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(
            inception_4b_double_3x3_reduce_bn_out
        )
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(
            inception_4b_relu_double_3x3_reduce_out
        )
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(
            inception_4b_double_3x3_1_out
        )
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(
            inception_4b_double_3x3_1_bn_out
        )
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(
            inception_4b_relu_double_3x3_1_out
        )
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(
            inception_4b_double_3x3_2_out
        )
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(
            inception_4b_double_3x3_2_bn_out
        )
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(
            inception_4b_pool_proj_out
        )
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(
            inception_4b_pool_proj_bn_out
        )
        inception_4b_output_out = torch.cat(
            [
                inception_4b_relu_1x1_out,
                inception_4b_relu_3x3_out,
                inception_4b_relu_double_3x3_2_out,
                inception_4b_relu_pool_proj_out,
            ],
            1,
        )
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(
            inception_4b_output_out
        )
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(
            inception_4c_3x3_reduce_out
        )
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(
            inception_4c_3x3_reduce_bn_out
        )
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(
            inception_4b_output_out
        )
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out
        )
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(
            inception_4c_double_3x3_reduce_bn_out
        )
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(
            inception_4c_relu_double_3x3_reduce_out
        )
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(
            inception_4c_double_3x3_1_out
        )
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(
            inception_4c_double_3x3_1_bn_out
        )
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(
            inception_4c_relu_double_3x3_1_out
        )
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(
            inception_4c_double_3x3_2_out
        )
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(
            inception_4c_double_3x3_2_bn_out
        )
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(
            inception_4c_pool_proj_out
        )
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(
            inception_4c_pool_proj_bn_out
        )
        inception_4c_output_out = torch.cat(
            [
                inception_4c_relu_1x1_out,
                inception_4c_relu_3x3_out,
                inception_4c_relu_double_3x3_2_out,
                inception_4c_relu_pool_proj_out,
            ],
            1,
        )
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(
            inception_4c_output_out
        )
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(
            inception_4d_3x3_reduce_out
        )
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(
            inception_4d_3x3_reduce_bn_out
        )
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(
            inception_4c_output_out
        )
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out
        )
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(
            inception_4d_double_3x3_reduce_bn_out
        )
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(
            inception_4d_relu_double_3x3_reduce_out
        )
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(
            inception_4d_double_3x3_1_out
        )
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(
            inception_4d_double_3x3_1_bn_out
        )
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(
            inception_4d_relu_double_3x3_1_out
        )
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(
            inception_4d_double_3x3_2_out
        )
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(
            inception_4d_double_3x3_2_bn_out
        )
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(
            inception_4d_pool_proj_out
        )
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(
            inception_4d_pool_proj_bn_out
        )
        inception_4d_output_out = torch.cat(
            [
                inception_4d_relu_1x1_out,
                inception_4d_relu_3x3_out,
                inception_4d_relu_double_3x3_2_out,
                inception_4d_relu_pool_proj_out,
            ],
            1,
        )
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(
            inception_4d_output_out
        )
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(
            inception_4e_3x3_reduce_out
        )
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(
            inception_4e_3x3_reduce_bn_out
        )
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(
            inception_4d_output_out
        )
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out
        )
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(
            inception_4e_double_3x3_reduce_bn_out
        )
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(
            inception_4e_relu_double_3x3_reduce_out
        )
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(
            inception_4e_double_3x3_1_out
        )
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(
            inception_4e_double_3x3_1_bn_out
        )
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(
            inception_4e_relu_double_3x3_1_out
        )
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(
            inception_4e_double_3x3_2_out
        )
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(
            inception_4e_double_3x3_2_bn_out
        )
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat(
            [
                inception_4e_relu_3x3_out,
                inception_4e_relu_double_3x3_2_out,
                inception_4e_pool_out,
            ],
            1,
        )
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(
            inception_4e_output_out
        )
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(
            inception_5a_3x3_reduce_out
        )
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(
            inception_5a_3x3_reduce_bn_out
        )
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(
            inception_4e_output_out
        )
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out
        )
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(
            inception_5a_double_3x3_reduce_bn_out
        )
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(
            inception_5a_relu_double_3x3_reduce_out
        )
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(
            inception_5a_double_3x3_1_out
        )
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(
            inception_5a_double_3x3_1_bn_out
        )
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(
            inception_5a_relu_double_3x3_1_out
        )
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(
            inception_5a_double_3x3_2_out
        )
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(
            inception_5a_double_3x3_2_bn_out
        )
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(
            inception_5a_pool_proj_out
        )
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(
            inception_5a_pool_proj_bn_out
        )
        inception_5a_output_out = torch.cat(
            [
                inception_5a_relu_1x1_out,
                inception_5a_relu_3x3_out,
                inception_5a_relu_double_3x3_2_out,
                inception_5a_relu_pool_proj_out,
            ],
            1,
        )
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(
            inception_5a_output_out
        )
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(
            inception_5b_3x3_reduce_out
        )
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(
            inception_5b_3x3_reduce_bn_out
        )
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(
            inception_5a_output_out
        )
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out
        )
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(
            inception_5b_double_3x3_reduce_bn_out
        )
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(
            inception_5b_relu_double_3x3_reduce_out
        )
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(
            inception_5b_double_3x3_1_out
        )
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(
            inception_5b_double_3x3_1_bn_out
        )
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(
            inception_5b_relu_double_3x3_1_out
        )
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(
            inception_5b_double_3x3_2_out
        )
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(
            inception_5b_double_3x3_2_bn_out
        )
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(
            inception_5b_pool_proj_out
        )
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(
            inception_5b_pool_proj_bn_out
        )
        inception_5b_output_out = torch.cat(
            [
                inception_5b_relu_1x1_out,
                inception_5b_relu_3x3_out,
                inception_5b_relu_double_3x3_2_out,
                inception_5b_relu_pool_proj_out,
            ],
            1,
        )
        return inception_5b_output_out

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        feat = x.clone()
        x = self.last_linear(x)
        return x, feat

    def forward(self, input):
        x = self.features(input)
        x, feat = self.logits(x)
        return x, feat


def bninception(in_channels, modality, pretrained="imagenet"):
    """
        BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper.
    """
    model = BNInception()
    if pretrained is not None:
        settings = pretrained_settings["bninception"][pretrained]
        model.load_state_dict(model_zoo.load_url(settings["url"]))
        model.input_space = settings["input_space"]
        model.input_size = settings["input_size"]
        model.input_range = settings["input_range"]
        model.mean = settings["mean"]
        model.std = settings["std"]

    if modality != "RGB":
        model.conv1_7x7_s2 = nn.Conv2d(
            in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )

    return model
