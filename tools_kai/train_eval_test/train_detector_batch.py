from __future__ import division

from mmcv import Config
from mmcv.runner import obj_from_dict
from mmdet import datasets, __version__
from mmdet.apis import (train_detector, get_root_logger)
from mmdet.models import build_detector
import os
import os.path as osp
import getpass
import time


"""
Author: WangCK
Date: 2019.12.11
Description: This script is used to train detectors with config file.
"""


def main():
    configs = \
        [
            # '../../configs/cvc09/faster_rcnn_r50_c4_cvc.py',
            # '../../configs/cvc09/faster_rcnn_r50_fpn_cvc.py',
            # '../../configs/cvc09/faster_rcnn_r50_mlfpn_cvc.py',
            # '../../configs/cvc09/faster_rcnn_v16_c5_cvc.py',
            # '../../configs/cvc09/faster_rcnn_v16_fpn_cvc.py',

            # Author:YY     dataset:kaist       backbone:r50
            '../../configs/kaist_fpn/mul_faster_rcnn_r50_fpn_add_kaist.py',

            # Author:WangCK     dataset:kaist       backbone:r50
            # '../../configs/kaist/mul_faster_rcnn_r50_mlfpn_add_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_r50_mlfpn_cat_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_r50_pre_mlfpn_add_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_r50_pre_mlfpn_cat_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_r50_c4_add_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_r50_c4_cat_kaist.py',


            # dataset:kaist   backbone:v16

        ]

    for config in configs:
        # load dataset
        cfg = Config.fromfile(config)
        cfg.gpus = 1

        if not osp.exists(cfg.work_dir):
            os.mkdir(cfg.work_dir)
        if cfg.checkpoint_config is not None:
            # save mmdet version in checkpoints as meta datasets
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__, config=cfg.text)

        username = getpass.getuser()
        temp_file = '/home/' + username + '/WangCK/Data/temp/temp.txt'
        # temp_file = '/media/' + username + '/3rd/WangCK/Data/temp/temp.txt'
        fo = open(temp_file, 'w+')
        # str_write = cfg.work_dir.replace('../..', ('/media/'+username+'/3rd/WangCK/Data'))
        str_write = cfg.work_dir.replace('../..', ('/home/' + username + '/WangCK/Data'))
        fo.write(str_write)
        fo.close()

        distributed = False
        # init logger before other steps
        logger = get_root_logger(cfg.log_level)     # cfg.log_level='INFO'
        logger.info("Distributed training: {}".format(distributed))

        # 1.build model
        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        # 2.create datasets used for train and validation
        train_dataset = obj_from_dict(cfg.data.train, datasets)
        # print(train_dataset.img_infos)       # debug

        t0 = time.time()        # the start training time
        # 3.train a detector
        train_detector(
            model,
            train_dataset,
            cfg,
            distributed=distributed,
            validate=True,
            logger=logger
        )
        t1 = time.time()        # the end training time
        logger.info("Total training time: {}m".format((t1-t0) // 60))
        logger.info("Total training time: {}minutes - {}hours".format((t1 - t0) // 60,
                                                                      (t1 - t0) // 3600))


if __name__ == '__main__':
    main()
