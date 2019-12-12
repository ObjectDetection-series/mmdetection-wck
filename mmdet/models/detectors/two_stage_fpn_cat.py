from .two_stage_mul import TwoStageDetectorMul
from ..registry import DETECTORS
import torch.nn as nn
import torch
from mmcv.cnn import kaiming_init

"""
Author: WangCK
Date: 2019.12.10
Description: This file defines class TwoStageDetectorFPNCat, inherited from TwoStageDetectorMul.
"""


@DETECTORS.register_module
class TwoStageDetectorFPNCat(TwoStageDetectorMul):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(TwoStageDetectorFPNCat, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        """
        Author: YY
        Description: These fusion layers are used to fusion FPNs from RGB channel and Thermal channel.
        """
        for i in range(4):
            conv_name = "conv{}".format(i)
            if backbone.type == 'MulResnet':
                self.add_module(conv_name, nn.Conv2d(512, 256, 1))
            elif backbone.type == 'MulVGG':
                self.add_module(conv_name, nn.Conv2d(256, 128, 1))
            kaiming_init(getattr(self, conv_name))
            # relu_name = "relu{}".format(i)
            # self.add_module(relu_name, nn.ReLU)

    def extract_feat(self, img, img_t):
        # extract feature maps of RGB channel and Thermal channel respectively
        feats_rgb, feats_t = self.backbone(img, img_t)
        # build FPN of RGBs and Thermal respectively
        if self.with_neck:
            x_rgb = self.neck(feats_rgb)
            x_t = self.neck(feats_t)
        x = []
        for i, (r, t) in enumerate(zip(x_rgb, x_t)):
            temp = torch.cat([r, t], 1)     # concatenate
            conv_name = "conv{}".format(i)
            conv_model = getattr(self, conv_name)
            out = conv_model(temp)

            # relu_name = "relu{}".format(i)
            # relu_model = getattr(self, relu_name)
            # out = relu_model(out)

            x.append(out)

        return tuple(x)
