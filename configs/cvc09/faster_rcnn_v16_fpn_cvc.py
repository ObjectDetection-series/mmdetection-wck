# model settings
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://vgg16',
    backbone=dict(
        type='VGG',
        depth=16,
        num_stages=5,
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        with_last_pool=True
    ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=128,
        out_indices=[0, 1, 2, 3],
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=128,
        feat_channels=128,
        anchor_scales=[8, 10, 12, 14],
        anchor_ratios=[1.0 / 0.5, 1.0],
        anchor_strides=[4, 8, 16, 32],
        anchor_base_sizes=[4, 8, 16, 32],       # ?
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        # use_sigmoid_cls=True
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=-1),
        out_channels=128,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=128,
        fc_out_channels=256,
        roi_feat_size=7,
        num_classes=2,  # background and pederstrian
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
            add_gt_as_proposals=False,
            pos_balance_sampling=False,
            neg_balance_thr=0),
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
data_root = '/home/server-248/WangCK/Data/datasets/CVC/'
img_norm_cfg = dict(
    mean=[123.675, 123.675, 123.675], std=[58.395, 58.395, 58.395], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 480), keep_ratio=True),    # diff
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
        img_scale=(640, 480),      # diff
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
    imgs_per_gpu=4,
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
        ipipeline=test_pipeline,
        test_mode=True))  # It is added by kai

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=2000,
    # warmup_ratio=1.0 / 3,
    step=[4, 8])
checkpoint_config = dict(interval=1)

# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../../work_dirs/faster_rcnn_v16_fpn_cvc'        # log文件和模型保存的路径
load_from = None
resume_from = None
workflow = [('train', 1)]
