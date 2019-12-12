from .two_stage_kai import TwoStageDetectorKai
from ..registry import DETECTORS

"""
Author: WangCK
Date: 2019.12.10
Description: This file defines class FasterRCNNKai inherited from class TwoStageDetectorKai.
"""


@DETECTORS.register_module
class FasterRCNNKai(TwoStageDetectorKai):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNNKai, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
