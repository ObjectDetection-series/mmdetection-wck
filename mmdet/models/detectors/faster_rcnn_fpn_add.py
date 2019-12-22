from .two_stage_fpn_add import TwoStageDetectorFPNAdd
from ..registry import DETECTORS

"""
Author: WangCK
Date: 2019.12.11
Description: This file defines class FasterRCNNMulFPNAdd, inherited from TwoStageDetectorFPNAdd.
"""


@DETECTORS.register_module
class FasterRCNNMulFPNAdd(TwoStageDetectorFPNAdd):

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
        super(FasterRCNNMulFPNAdd, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
