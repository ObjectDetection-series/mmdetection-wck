import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch.nn as nn
import pycocotools.mask as maskUtils


from mmdet.core import tensor2imgs, get_classes

"""
Author: WangCK
Date: 2019.12.10
Description: This file defines a Multispectral base detector, namely class BaseDetectorMul.
"""


class BaseDetectorMul(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetectorMul, self).__init__()
        self.fp16_enabled = False  # diff

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property

    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs, imgs_t):
        pass

    def extract_feats(self, imgs, imgs_t):
        assert isinstance(imgs, list)
        assert isinstance(imgs_t, list)
        for img_rgb, img_t in zip(imgs, imgs_t):
            yield self.extract_feat(img_rgb, imgs_t)

    @abstractmethod
    def forward_train(self, imgs, imgs_t, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_t, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, imgs_t, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, imgs_t, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (imgs_t, 'imgs_t'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))

        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], imgs_t[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, imgs_t, img_metas, **kwargs)

    def forward(self, img, img_t, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_t, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_t, img_meta, **kwargs)

    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset='coco',
                    score_thr=0.3):
        img_tensor = data['img'][0]
        # img_tensor_t = datasets['img_t'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        # imgs_t = tensor2imgs(img_tensor_t,**)
        assert len(imgs) == len(img_metas)

        if isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, list):
            class_names = dataset
        else:
            raise TypeError('dataset must be a valid dataset name or a list'
                            ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(result)
            ]
            labels = np.concatenate(labels)
            bboxes = np.vstack(result)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr)






