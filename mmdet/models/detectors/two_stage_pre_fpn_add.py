from .two_stage_mul import TwoStageDetectorMul
from ..registry import DETECTORS

"""
Author: WangCK
Date: 2019.12.11
Description: This file defines class TwoStageDetectorPreFPNAdd, inherited from TwoStageDetectorMul.
"""


@DETECTORS.register_module
class TwoStageDetectorPreFPNAdd(TwoStageDetectorMul):

    """
    (1) Fuse feature maps by element-wise adding
    (2) Build FPN based on fused feature maps
    """
    def extract_feat(self, img, img_t):
        # extract feature maps of RGB channel and Thermal channel
        feats_rgb, feats_t = self.backbone(img, img_t)
        x = []
        for (r, t) in zip(feats_rgb, feats_t):
            temp = r + t        # element-wise add in each feature maps of RGB images and Thermal images
            x.append(temp)

        # build FPN/MLFPN/FPN+BFP based on fused feature maps
        x = tuple(x)
        if self.with_neck:
            x = self.neck(x)
        return tuple(x)
