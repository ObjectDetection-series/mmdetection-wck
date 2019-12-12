from .two_stage_mul import TwoStageDetectorMul
from ..registry import DETECTORS

"""
Author: WangCK
Date: 2019.12.10
Description: This file defines class FasterRCNNMul, inherited from TwoStageDetectorMul.
"""


@DETECTORS.register_module
class FasterRCNNMul(TwoStageDetectorMul):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNNMul, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
