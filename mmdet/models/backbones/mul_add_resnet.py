import torch.nn as nn
import torch
from .resnet import ResNet
from ..registry import BACKBONES

"""
Author: WangCK
Date: 2019.12.10
Description: This file defines a ResNet to process multi-modal datasets.
             Standard ResNet will process each modal respectively.
             Then, the results from these standard ResNets will be added in element-wise.
"""

@BACKBONES.register_module
class MulAddResnet(nn.Module):

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),  # diff with YY: normalize=dict(type='BN', frozen=False)
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 with_cp=False,
                 zero_init_residual=True):
        super(MulAddResnet, self).__init__()
        self.resnet_rgb = ResNet(
            depth=depth,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual
        )

        # ResNet used for processing thermal images(thermal images should be expanded to three channels)
        self.resnet_thermal = ResNet(
            depth=depth,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual
        )

    def forward(self, img_rgb, img_th):
        out_rgb = self.resnet_rgb(img_rgb)
        out_th = self.resnet_thermal(img_th)
        assert len(out_rgb) == len(out_th)
        x = []
        for i, (r, t) in enumerate(zip(out_rgb, out_th)):
            out = r + t       # element-wise add features from two sibling branches
            x.append(out)
        return tuple(x)

    def init_weights(self, pretrained=None):
        self.resnet_rgb.init_weights(pretrained)
        self.resnet_thermal.init_weights(pretrained)

    def train(self, model=True):
        self.resnet_rgb.train(model)
        self.resnet_thermal.train(model)
