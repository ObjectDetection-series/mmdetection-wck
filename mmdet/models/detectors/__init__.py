from .base import BaseDetector
from .faster_rcnn import FasterRCNN
from .rpn import RPN
from .single_stage import SingleStageDetector
from .two_stage import TwoStageDetector

"""
Kai add following lines.
"""
from .base_mul import BaseDetectorMul
from .two_stage_mul import TwoStageDetectorMul
from .faster_rcnn_mul import FasterRCNNMul
from .faster_rcnn_fpn_add import TwoStageDetectorFPNAdd
from .faster_rcnn_fpn_cat import FasterRCNNMulFPNCat
from .two_stage_pre_fpn_add import TwoStageDetectorPreFPNAdd
from .two_stage_pre_fpn_cat import TwoStageDetectorPreFPNCat
from .faster_rcnn_pre_fpn_add import FasterRCNNMulPreFPNAdd
from .faster_rcnn_pre_fpn_cat import FasterRCNNMulPreFPNCat


"""
The following files are not used, so deleted.
If need later, copy from "mmdetection".
"""
# from .cascade_rcnn import CascadeRCNN   # del
# from .double_head_rcnn import DoubleHeadRCNN        # del
# from .fast_rcnn import FastRCNN # del
# from .fcos import FCOS  # del
# from .fovea import FOVEA    # del
# from .grid_rcnn import GridRCNN # del
# from .htc import HybridTaskCascade  # del
# from .mask_rcnn import MaskRCNN # del
# from .mask_scoring_rcnn import MaskScoringRCNN  # del
# from .reppoints_detector import RepPointsDetector   # del
# from .retinanet import RetinaNet    # del


__all__ = [
    'BaseDetector', 'SingleStageDetector', 'TwoStageDetector', 'RPN',
    'FasterRCNN',

    'BaseDetectorMul', 'TwoStageDetectorMul', 'FasterRCNNMul', 'TwoStageDetectorFPNAdd',
    'FasterRCNNMulFPNCat', 'TwoStageDetectorPreFPNAdd', 'TwoStageDetectorPreFPNCat',
    'FasterRCNNMulPreFPNAdd', 'FasterRCNNMulPreFPNCat'
]
