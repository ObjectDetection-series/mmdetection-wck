from .two_stage_pre_fpn_add import TwoStageDetectorPreFPNAdd
from ..registry import DETECTORS


"""
Author: WangCK
Date: 2019.12.11
Description: This file defines class FasterRCNNMulPreFPNAdd, inherited from TwoStageDetectorPreFPNAdd.
"""


@DETECTORS.register_module
class FasterRCNNMulPreFPNAdd(TwoStageDetectorPreFPNAdd):

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
        super(FasterRCNNMulPreFPNAdd, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
