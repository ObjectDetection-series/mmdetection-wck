# model settings
# input_size = 320        # this param is for VGG
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='MLFPN',
        backbone_choice="ResNet",   # "SSD" or "ResNet"
        in_channels=[256, 512, 1024, 2048],
        out_indices=[0, 1, 2, 3],   # kai add the param, correspond to 'scale_outs_num'(not used!)
        planes=256,                 # out_channels of each scale feature
        scale_outs_num=4,           # the num of scale obtained by each TUM
        tum_num=2,                  # the num of TUM module: 8 -> 4 -> 2
        smooth=True,                # the param is used in TUM
        base_feature_size=4,        # ?
        base_choice=2,              # the param is used to choose 'ResNet' or others
        base_list=[2, 3],           # the param is used to choose the elements in 'dim_conv'
        norm=True,
        ssd_style_tum=False         # ?
        # the size of the smallest tum ouput =>( '-2' or '/2')
    ),                              # backbone + MFLPN -> [4, 2048, H, W]
    rpn_head=dict(
        type='RPNHead',
        in_channels=512,            # kai change this param: 256 -> 2048 -> 1024 -> 512
        feat_channels=256,          # kai change this param: 128 -> 256
        anchor_scales=[8, 10, 12, 14],
        anchor_ratios=[1.0 / 0.5, 1.0],
        anchor_strides=[4, 8, 16, 32],
        anchor_base_sizes=[4, 8, 16, 32],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        # use_sigmoid_cls=True),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=-1),
        out_channels=512,             # kai: 256 -> 2048 -> 1024 -> 512
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=512,        # kai: 256 -> 2048 -> 1024 -> 512
        fc_out_channels=512,    # kai: 512
        roi_feat_size=7,
        num_classes=2,      # background and pedestrian
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=120,
            pos_fraction=1.0 / 4,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
        nms=dict(
            nms_across_levels=False,
            nms_pre=20000,
            nms_post=20000,
            max_num=5000,
            nms_thr=0.9,
            min_bbox_size=0)),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=64,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=10000,
        nms_post=10000,
        max_num=300,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.1, nms=dict(type='nms', iou_thr=0.5), max_per_img=40))

# dataset settings
# dataset_type = 'ExtendedCvcDataset'
dataset_type = 'CvcDataset'
data_root = '/home/ser248/WangCK/Data/datasets/CVC/'
img_norm_cfg = dict(
    mean=[123.675, 123.675, 123.675], std=[58.395, 58.395, 58.395], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(960, 736), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 736),      # diff
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,     # batch size
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations-pkl/train-all.pkl',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations-pkl/test-all.pkl',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations-json/test-all.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)    # lr: 0.001 -> 0.01 -> 0.02 weight_decay: 0.0001
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=2000,
    # warmup_ratio=1.0 / 3,
    step=[4, 8])
checkpoint_config = dict(interval=5)

# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# runtime settings
total_epochs = 130       # 20 -> 40 -> 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = '../../work_dirs/faster_rcnn_r50_fpn_cvc'        # log文件和模型保存的路径
work_dir = '/media/ser248/3rd/WangCK/Data/mmdetection-wck/work_dirs/faster_rcnn_r50_mlfpn_cvc'
load_from = None
resume_from = '/media/ser248/3rd/WangCK/Data/mmdetection-wck/work_dirs/faster_rcnn_r50_mlfpn_cvc/latest.pth'
workflow = [('train', 1)]
