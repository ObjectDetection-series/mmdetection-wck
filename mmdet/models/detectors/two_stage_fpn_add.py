from .two_stage_mul import TwoStageDetectorMul
from ..registry import DETECTORS


"""
Author: WangCK
Date: 2019.12.11
Description: This file defines class TwoStageDetectorFPNAdd, inherited from TwoStageDetectorMul.
"""


@DETECTORS.register_module
class TwoStageDetectorFPNAdd(TwoStageDetectorMul):

    def extract_feat(self, img, img_t):
        # extract feature maps of RGB channel and Thermal respectively
        feats_rgb, feats_t = self.backbone(img, img_t)
        # build FPN of RGB and Thermal respectively
        if self.with_neck:
            x_rgb = self.neck(feats_rgb)
            x_t = self.neck(feats_t)
        x = []
        for (r, t) in zip(x_rgb, x_t):
            temp = r + t        # element-wise add in each layer in FPN
            x.append(temp)
        return tuple(x)
