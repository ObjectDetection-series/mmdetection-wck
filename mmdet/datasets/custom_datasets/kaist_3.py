from ..customV056 import CustomDatasetV056
from ..registry import DATASETS
import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from ..utils import to_tensor, random_scale, to_tensor, random_scale
from ..transforms import (ImageTransform, BboxTransform, MaskTransform,
                          Numpy2Tensor)
import cv2


@DATASETS.register_module
class KaistDataset(CustomDatasetV056):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 img_norm_cfg_t,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False):
        self.img_norm_cfg_t = img_norm_cfg_t
        # transforms
        self.img_transform_t = ImageTransform(
            size_divisor=size_divisor, **self.img_norm_cfg_t)
        super(KaistDataset, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            img_scale=img_scale,
            img_norm_cfg=img_norm_cfg,
            size_divisor=size_divisor,
            proposal_file=proposal_file,
            num_max_proposals=num_max_proposals,
            flip_ratio=flip_ratio,
            with_mask=with_mask,
            with_crowd=with_crowd,
            with_label=with_label,
            test_mode=test_mode)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image(rgb)
        img_temp = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if img_temp.shape[2] == 1:
            img = np.zeros((img_temp.shape[0], img_temp.shape[1], 3))
            img[:, :, 0] = img_temp
            img[:, :, 1] = img_temp
            img[:, :, 2] = img_temp
        else:
            img = img_temp
        """
        kai: augment the thermal image with corresponding saliency map. Do this by replacing
        one duplicate channel of the 3-channel thermal images with corresponding saliency map.
        """
        # load original thermal image
        original_t_path = osp.join(self.img_prefix, img_info['filename']).replace('visible', 'lwir')
        # original_t = cv2.imread(original_t_path)

        # load saliency map
        saliency_map_path = original_t_path.replace('images', 'saliencyMaps/train_masks')
        saliency_map = cv2.imread(saliency_map_path)

        # augment original image with saliency map
        # img_t = np.zeros_like(saliency_map)
        # img_t[:, :, 0] = saliency_map[:, :, 0]
        # img_t[:, :, 1] = saliency_map[:, :, 1]
        # img_t[:, :, 2] = saliency_map[:, :, 2]
        img_t = saliency_map

        # 显示Fused thermal image
        # screen_res = 640, 512
        # scale_width = screen_res[0] / img_t.shape[1]
        # scale_height = screen_res[1] / img_t.shape[0]
        # scale = min(scale_width, scale_height)
        # window_width = int(img_t.shape[1] * scale)
        # window_height = int(img_t.shape[0] * scale)
        # cv2.namedWindow('Fused Thermal Iamge', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Fused Thermal Iamge', window_width, window_height)

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        # for rgb images
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip)
        # for thermal images
        img_t, img_shape_t, pad_shape_t, scale_factor_t = self.img_transform_t(
            img_t, img_scale, flip)
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_t=DC(to_tensor(img_t), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))

        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img_temp = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if img_temp.shape[2] == 1:
            img = np.zeros((img_temp.shape[0], img_temp.shape[0], ))
            img[:, :, 0] = img_temp
            img[:, :, 1] = img_temp
            img[:, :, 2] = img_temp
        else:
            img = img_temp
        # load original thermal image
        original_t_path = osp.join(self.img_prefix, img_info['filename']).replace('visible', 'lwir')
        # original_t = cv2.imread(original_t_path)

        # load saliency map
        saliency_map_path = original_t_path.replace('images', 'saliencyMaps/test_masks')
        saliency_map = cv2.imread(saliency_map_path)

        # augment original image with saliency map
        # img_t = np.zeros_like(saliency_map)
        # img_t[:, :, 0] = saliency_map[:, :, 0]
        # img_t[:, :, 1] = saliency_map[:, :, 1]
        # img_t[:, :, 2] = saliency_map[:, :, 2]
        img_t = saliency_map

        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, img_t, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip)
            _img_t, img_shape, pad_shape, scale_factor = self.img_transform_t(
                img_t, scale, flip)
            _img = to_tensor(_img)
            _img_t = to_tensor(_img_t)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_t, _img_meta, _proposal

        imgs = []
        imgs_t = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_t, _img_meta, _proposal = prepare_single(
                img, img_t, scale, False, proposal)
            imgs.append(_img)
            imgs_t.append(_img_t)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_t, _img_meta, _proposal = prepare_single(
                    img, img_t, scale, True, proposal)
                imgs.append(_img)
                imgs_t.append(_img_t)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_t=imgs_t, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
