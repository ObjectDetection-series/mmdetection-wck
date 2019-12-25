# model settings
model = dict(
    type='FasterRCNN',      # model类型
    pretrained='torchvision://resnet50',    # 预训练模型
    backbone=dict(
        type='ResNet',                  # backbone类型
        depth=50,                       # 网络层数
        num_stages=4,                   # resnet的stage数量
        out_indices=(0, 1, 2, 3),       # 输出的stage序号
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',                     # neck类型
        in_channels=[256, 512, 1024, 2048],     # 输入的各个stage的通道数
        out_channels=256,                       # 输出的特征层的通道数
        out_indices=[0, 1, 2, 3],               # YY add the param
        num_outs=4),                            # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',                         # RPN网络类型
        in_channels=256,                        # RPN网络的输入通道数
        feat_channels=128,                      # 特征层的通道数
        anchor_scales=[8, 10, 12, 14],          # 生成的anchor的baselen，baselen = sqrt(w*h)，w和h为anchor的宽和高
        anchor_ratios=[1.0 / 0.5, 1.0],         # anchor的宽高比
        anchor_strides=[4, 8, 16, 32],          # 在每个特征层上的anchor的步长（对应于原图）
        anchor_base_sizes=[4, 8, 16, 32],
        target_means=[.0, .0, .0, .0],          # 均值
        target_stds=[1.0, 1.0, 1.0, 1.0],       # 方差
        # use_sigmoid_cls=True),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',              # RoIExtractor类型
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=-1),
        out_channels=256,                       # 输出通道数
        featmap_strides=[4, 8, 16, 32]),        # 特征图的步长
    bbox_head=dict(
        type='SharedFCBBoxHead',                # FC类型
        num_fcs=2,                              # FC数量
        in_channels=256,                        # 输入通道数
        fc_out_channels=512,                    # 输出通道数
        roi_feat_size=7,
        num_classes=2,      # background and pedestrian
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        #  是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，
        #  后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',      # RPN网络的正负样本分配器类型
            pos_iou_thr=0.5,            # 正样本的iou阈值 - 可调参
            neg_iou_thr=0.3,            # 负样本的iou阈值 - 可调参
            # 正样本的iou最小值。如果assign给gt的anchors中最大的IOU低于0.3，
            # 则忽略所有的anchors，否则保留最大IOU的anchor - 可调参
            min_pos_iou=0.3,
            ignore_iof_thr=-1),         # 忽略bbox的阈值，当gt中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',       # 正负样本提取器类型
            num=120,                    # 需要提取的正负样本数量
            pos_fraction=1.0 / 4,       # 正样本比例, 0.25 - 可调参
            neg_pos_ub=-1,              # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False),     # 把gt加入proposal作为正样本
        allowed_border=0,               # 允许在bbox周围外扩一定的像素
        pos_weight=-1,                  # 正样本权重，-1表示不改变原始的权重
        debug=False,                    # debug模式
        nms=dict(      # 可调参，nms参数很重要
            nms_across_levels=False,    # 在所有的fpn层内做nms
            nms_pre=20000,              # 在nms之前，保留的得分最高的proposal数量
            nms_post=20000,             # 在nms之后，保留的的得分最高的proposal数量
            max_num=5000,               # 在后处理完成之后，保留的proposal数量
            nms_thr=0.9,                # nms阈值
            min_bbox_size=0)),          # 最小bbox尺寸
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
    rpn=dict(                       # 测试时的RPN参数
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
# 输入图像初始化，减去均值mean，再除以方差std，to_rgb表示将bgr转为rgb
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
        img_scale=(640, 480),
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
    imgs_per_gpu=2,         # 每个gpu计算的图像数量
    workers_per_gpu=2,      # 每个gpu分配的线程数
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations-pkl/train-all.pkl',   # 数据集annotation的路径
        img_prefix=data_root + 'images/',       # 数据集图片的路径
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
# 优化参数，lr学习率，momentum动量因子，weight_decay权重衰减因子 - 可调参
# 一般，当gpu数量为8时,lr=0.02；当gpu数量为4时,lr=0.01；我只有一个gpu，所以设置lr=0.0025，修改前是0.02
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))       # 梯度均衡参数

# learning policy
lr_config = dict(
    policy='step',                      # 优化策略
    # warmup='linear',                  # 初始的学习率增加的策略，linear为线性增加
    # warmup_iters=2000,                # 在初始的2000次迭代中学习率逐渐增加
    # warmup_ratio=1.0 / 3,             # 起始的学习率?
    step=[4, 8])                        # 在第4和8个epoch时降低学习率 - 可调参
checkpoint_config = dict(interval=1)    # 间隔多少个epoch，保存一次模型

# yapf:disable
log_config = dict(
    interval=1000,                      # 每1000个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),    # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# runtime settings
total_epochs = 20                       # 最大epoch数 - 可调参
dist_params = dict(backend='nccl')      # 分布式参数
log_level = 'INFO'                      # 输出信息的完整度级别
work_dir = '../../work_dirs/faster_rcnn_r50_fpn_cvc'        # log文件和模型保存的路径
load_from = None                        # 加载模型的路径，None表示从预训练模型加载
resume_from = None                      # 恢复训练模型的路径
workflow = [('train', 1)]
