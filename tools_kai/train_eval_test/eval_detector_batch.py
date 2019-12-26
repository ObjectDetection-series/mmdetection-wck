import torch
import mmcv
from mmcv.runner import load_checkpoint, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet import datasets
from mmdet.core import eval_cvc_mr, eval_caltech_mr, eval_kaist_mr
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector
import os.path as osp
import numpy as np
import os
import getpass


"""
Author: WangCK
Date: 2019.12.12
Description: This script is used to eval pre-trained detectors.
             Add eval_detector_batch.py. It is used to eval pre-trained detectors.
"""


def single_test(model, data_loader, show=False):
    model.eval()
    results = []        # empty list
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
            results.append(result)

        # for test
        img_path = data_loader.dataset.img_infos[i]['filename']
        file_path = img_path.replace('images', 'res')
        if 'visible' in file_path:
            file_path = file_path.replace('/visible/', '/')
        file_path = file_path.replace('.jpg', '.txt')
        if os.path.exists(file_path):
            os.remove(file_path)
        os.mknod(file_path)     # os.mknod()：用于创建一个指定文件名的文件系统节点
        """
        For faster-rcnn, the result is a list, each element in list is result for a object class.
        In pedestrian detection, there is only one class.
        For RPN, the result is a numpy. The result of RPN is category-independent.
        """
        if isinstance(results, list):
            np.savetxt(file_path, result[0])
        else:
            np.savetxt(file_path, result)

        if show:
            model.module.show_result(data, result,
                                     data_loader.dataset.img_norm_cfg, dataset=['person'], score_thr=0.1)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def main():
    configs = [
        # '../../configs/cvc09/faster_rcnn_r50_c4_cvc.py',
        # '../../configs/cvc09/faster_rcnn_r50_fpn_cvc.py',
        # '../../configs/cvc09/faster_rcnn_r50_mlfpn_cvc.py',
        # '../../configs/cvc09/faster_rcnn_v16_c5_cvc.py',
        # '../../configs/cvc09/faster_rcnn_v16_fpn_cvc.py',

        # Author:YY     dataset:kaist_mlfpn       backbone:r50
        # '../../configs/kaist_fpn/mul_faster_rcnn_r50_fpn_add_kaist.py',

        # Author:WangCK     dataset:kaist_mlfpn       backbone:r50
        # '../../configs/kaist_mlfpn/mul_faster_rcnn_r50_mlfpn_add_kaist.py',
        # '../../configs/kaist_mlfpn/mul_faster_rcnn_r50_mlfpn_cat_kaist.py',
        # '../../configs/kaist_mlfpn/mul_faster_rcnn_r50_pre_mlfpn_add_kaist.py',
        # '../../configs/kaist_mlfpn/mul_faster_rcnn_r50_pre_mlfpn_cat_kaist.py',
        # '../../configs/kaist_mlfpn/mul_faster_rcnn_r50_c4_add_kaist.py',
        # '../../configs/kaist_mlfpn/mul_faster_rcnn_r50_c4_cat_kaist.py',

        # Author:WangCK  dataset:kaist_[backbone:r50 + neck:BFP]
        # '../../configs/kaist_bfp/mul_libra_faster_rcnn_r50_fpn_add_kaist_YY.py',
        '../../configs/kaist_bfp/mul_libra_faster_rcnn_r50_fpn_add_kaist_1.py'

        # Author:WangCK  dataset:kaist_[backbone:v16 + neck:BFP]
    ]
    for config in configs:
        # load dataset
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        username = getpass.getuser()
        # temp_file = '/home/' + username + '/WangCK/Data/temp/temp.txt'
        temp_file = '/media/' + username + '/3rd/WangCK/Data/temp/temp.txt'
        fo = open(temp_file, 'w+')
        # str_write = cfg.work_dir.replace('../..', ('/home/' + username + '/WangCK/Data'))
        str_write = cfg.work_dir.replace('../..', ('/media/'+username+'/3rd/WangCK/Data'))
        fo.write(str_write)
        fo.close()

        dataset = obj_from_dict(cfg.data.val, datasets, dict(test_mode=True))
        # load model
        checkpoint_file = osp.join(cfg.work_dir, 'epoch_26.pth')
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, checkpoint_file)
        model = MMDataParallel(model, device_ids=[0])

        # build dataloader
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)

        # eval outputs
        outputs = single_test(model, data_loader, True)
        if 'caltech' in config:
            eval_caltech_mr()
        if 'kaist' in config:
            eval_kaist_mr()
        if 'cvc' in config:
            eval_cvc_mr()


if __name__ == '__main__':
    main()
