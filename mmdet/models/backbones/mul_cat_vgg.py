import torch.nn as nn
import torch
from .vgg import VGG
from ..registry import BACKBONES
from mmcv.cnn import kaiming_init

"""
Author: WangCK
Date: 2019.12.10
Description: This file defines a VGGNet to process multi-modal(two) datasets.
             Standard VGGNet will process each modal respectively.
             Then, the results from these standard ResNets will be concatenated. 
"""

@BACKBONES.register_module
class MulCatVGG(nn.Module):

    def __init__(self,
                 depth,
                 with_bn=False,
                 num_classes=-1,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 ceil_mode=False,
                 with_last_pool=True
                 ):
        super(MulCatVGG, self).__init__()
        self.vgg_rgb = VGG(
            depth=depth,
            with_bn=with_bn,
            num_classes=num_classes,
            num_stages=num_stages,
            dilations=dilations,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            bn_eval=bn_eval,
            bn_frozen=bn_frozen,
            ceil_mode=ceil_mode,
            with_last_pool=with_last_pool
        )
        self.vgg_thermal = VGG(
            depth=depth,
            with_bn=with_bn,
            num_classes=num_classes,
            num_stages=num_stages,
            dilations=dilations,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            bn_eval=bn_eval,
            bn_frozen=bn_frozen,
            ceil_mode=ceil_mode,
            with_last_pool=with_last_pool
        )
        for i in out_indices:
            conv_name = "conv{}".format(i)
            if i > 3:
                i = 3
            self.add_module(conv_name, nn.Conv2d(int(128 * 2 ** i), int(64 * 2 ** i), 1))
            kaiming_init(getattr(self, conv_name))
            # relu_name = "relu{}".format{i}
            # self.add_module(nn.ReLU)
        self.out_indices = out_indices

    def forward(self, img_rgb, img_th):
        out_rgb = self.vgg_rgb(img_rgb)
        out_th = self.vgg_thermal(img_th)
        assert len(out_rgb) == len(out_th)
        x = []
        for i, (r, t) in enumerate(zip(out_rgb, out_th)):
            temp = torch.cat([r, t], 1)
            conv_name = "conv{}".format(self.out_indices[i])
            conv_model = getattr(self, conv_name)
            out = conv_model(temp)      # concatenate features from two sibling branches
            # relu_name = "relu{}".format(i)
            # relu_model = getattr(self, relu_name)
            # out = relu_model(out)
            x.append(out)
        return tuple(x)

    def init_weights(self, pretrained=None):
        self.vgg_rgb.init_weights(pretrained)
        self.vgg_thermal.init_weights(pretrained)

    def train(self, model=True):
        self.vgg_rgb.train(model)
        self.vgg_thermal.train(model)
