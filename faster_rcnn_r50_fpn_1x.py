"""
配置文件是一个字典，每个key的value值也可能是个字典
  主要内容                                   key
- model settings                            model
- model training and testing settings       train_cfg / test_cfg
- dataset setting                           data_root / img_norm_cfg / train_pipeline / test_pipeline / data
- optimizer                                 optimizer / optimizer_config
- learning policy                           lr_config / checkpoint_config
- yapf:disable/enbale                       log_config
- runtime settings                          total_epochs / dist_params / log_level / work_dir / load_from / resume_from / workflow
"""

# model settings
model = dict(
    type='FasterRCNN',      # model类型
    pretrained='torchvision://resnet50',    # 预训练模型
    backbone=dict(
        type='ResNet',      # backbone类型
        depth=50,           # 网络深度
        num_stages=4,       # resnet的stage数量
        out_indices=(0, 1, 2, 3),   # 输出的stage序号
        frozen_stages=1,            # 冻结的stage数量，既该state不更新参数，-1表示所有的stage都更新参数
        style='pytorch'),    # 如果设置为pytorch，则stride为2的层是conv3*3的卷积层；如果设置caffe，则stride为２的层是第一个conv１＊１的卷基层
    neck=dict(
        type='FPN',     # neck类型
        in_channels=[256, 512, 1024, 2048],     #　输入的各个stage的通道数
        out_channels=256,       # 输出的特征层的通道数
        num_outs=5),            # 输出的特征层的数量?
    rpn_head=dict(
        type='RPNHead',     # RPN网络类型
        in_channels=256,    # RPN网络的输入通道数
        feat_channels=256,  # 特征层的通道数
        anchor_scales=[8],  # 生成的anchor的baselen，baselen = sqrt(w*h)，w和h为anchor的宽和高
        anchor_ratios=[0.5, 1.0, 2.0],          # anchor的宽高比
        anchor_strides=[4, 8, 16, 32, 64],      # 在每个特征层上anchor的步长
        target_means=[.0, .0, .0, .0],          # 均值
        target_stds=[1.0, 1.0, 1.0, 1.0],       # 方差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',      # RoIExtractor类型
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),  # RoI具体参数：RoI类型为RoIAlign，输出尺寸为７，sample数为２
        out_channels=256,       # out-ch
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',    # 全连接层类型
        num_fcs=2,                  # 全连接层数量？
        in_channels=256,            # in-ch
        fc_out_channels=1024,       # out-ch
        roi_feat_size=7,            # RoI特征层尺寸
        num_classes=81,             # 类别数，COCO数据集的类别数+1
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',      # RPN网络的正负样本划分
            pos_iou_thr=0.7,            # pos:positive　正样本的IOU阈值
            neg_iou_thr=0.3,            # neg:negative　负样本的IOU阈值
            min_pos_iou=0.3,            # 正样本的最小IOU，如果assign给GT的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),         # 忽略bbox的阈值，当GT中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',       # 正负样本提取器类型
            num=256,                    # 需提取的正负样本数量
            pos_fraction=0.5,           # 正样本比例
            neg_pos_ub=-1,              # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False),     # 把GT加入proposal作为正样本
        allowed_border=0,               # 允许在bbox周围外扩一定的像素
        pos_weight=-1,                  # 正样本权重，-1表示不改变原始的权重
        debug=False),                   # 是否使用debug模式
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(       # 推断时的RPN参数
        nms_across_levels=False,    # 在所有的FPN层内做NMS
        nms_pre=1000,       # 在NMS之前保留的得分最高的proposal数量
        nms_post=1000,      # 在NMS之后保留的得分最高的proposal数量
        max_num=1000,       # 在后处理完成之后保留的proposal数量
        nms_thr=0.7,        # NMS阈值
        min_bbox_size=0),   # 最小的bbox尺寸
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)

# dataset settings
dataset_type = 'CocoDataset'    # 数据集类型
data_root = 'data/coco/'        # 数据集根目录
img_norm_cfg = dict(            # 输入图像初始化，减去mean，除以std，to_rgb表示：将bgr转为rgb
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
    imgs_per_gpu=2,         # 每个GPU计算的图像数量
    workers_per_gpu=2,      # 每个GPU分配的线程数量
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',    # 数据集annotations路径
        img_prefix=data_root + 'train2017/',    # 数据集的图片路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# 我只有一个GPU，所以设置lr=0.0025，修改前是0.02
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))       # 梯度均衡参数

# learning policy
lr_config = dict(           # 基准学习率为0.02，对于FPN，在15万次和45万次后除以１０
    policy='step',          # 优化策略
    warmup='linear',        # 出事的学习率增加的策略，linear为线性增长
    warmup_iters=500,       # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 3,   # 起始的学习率？
    step=[8, 11])           # 在第8、11个epoch时降低学习率
checkpoint_config = dict(interval=1)    # 间隔1epoch，保存一次模型

# yapf:disable
log_config = dict(
    interval=50,            # 每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),    # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# runtime settings
total_epochs = 12       # 最大epoch数，60万/7500次，7500=15000/2
dist_params = dict(backend='nccl')      # 分布式参数
log_level = 'INFO'      # 输出日志信息的完整度级别
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'     # log文件和模型文件存储路径
load_from = None        # 加载模型的路径，None表示从预训练模型加载
resume_from = None      # 恢复训练模型的路径
workflow = [('train', 1)]       # 当前工作区名称
