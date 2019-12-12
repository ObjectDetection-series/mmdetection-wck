import torch.nn as nn
from .vgg import VGG
from ..registry import BACKBONES

"""
Author:WangCK
Date:2019.12.9
Description: This file defines a VGGNet to process multi-model(two) datasets.
             Standard ResNet will process each model respectively. 
             Then, the results from these standard ResNets will be returned.
"""


@BACKBONES.register_module
class MulVGG(nn.Module):

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
        super(MulVGG, self).__init__()
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

    def forward(self, img_rgb, img_th):
        out_rgb = self.vgg_rgb(img_rgb)
        out_th = self.vgg_thermal(img_th)
        assert len(out_rgb) == len(out_th)
        return out_rgb, out_th

    def init_weights(self, pretrained=None):
        self.vgg_rgb.init_weights(pretrained)
        self.vgg_thermal.init_weights(pretrained)

    def train(self, model=True):
        self.vgg_rgb.train(model)
        self.vgg_thermal.train(model)
