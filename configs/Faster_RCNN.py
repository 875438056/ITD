# 继承自官方的 Faster R-CNN 基础配置
custom_imports = dict(
    imports=['models.backbones.YOLO_SPPF',
             'models.necks.identity_neck',
             'models.backbones.lightweight_hrnet_p2',
             'models.necks.PAN_neck_p2',
             'models.visualizers.custom_visualizer',
             'models.transforms.centroid_transforms',
             'models.activations.Mish',
             'models.heads.centroid_adaptive_heatmap_ltrb_offset_head',
             'models.transforms.custom_packer',
             'models.metrics.CocoMetricTinySize',
             'models.losses.KL_Loss',], allow_failed_imports=False)

_base_ = [
    r'F:\ITD\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_1x_coco.py'
]

# --- 1. 修改模型中的类别数量 ---
# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
        std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        init_cfg=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# --- 2. 修改数据集和数据加载器 ---
dataset_type = 'CocoDataset'
data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/' # 数据集根目录

# 定义数据集的类别元信息
metainfo = {
    'classes': ('target'),
    # 你可以为每个类别定义一个可视化颜色 (BGR格式)
    # 'palette': [(220, 20, 60), (119, 11, 32), ...]
}

# 修改训练数据加载器
train_dataloader = dict(
    batch_size=10,  # 根据你的 GPU 显存大小调整
    num_workers=2, # 根据你的 CPU 核心数调整
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/')
    )
)

# 修改验证数据加载器
val_dataloader = dict(
    batch_size=8,  # 根据你的 GPU 显存大小调整
    num_workers=2,  # 根据你的 CPU 核心数调整
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')
    )
)

# 修改测试数据加载器
test_dataloader = val_dataloader

# --- 3. 修改评估方式 ---
val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json'
)
test_evaluator = val_evaluator

# 训练、验证、测试的循环配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=140, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器封装
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='Adam', lr=0.0005),
#     clip_grad=dict(max_norm=35, norm_type=2))

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        by_epoch=True,
        milestones=[90, 120],  # 在第90和120个epoch降低学习率
        gamma=0.1)
]

default_scope = 'mmdet' # 默认的代码库作用域
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='auto'), # 每10个epoch保存一次权重
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',
                    draw=True,                 # 启用绘图
                    interval=50,               # 每50次iter保存一次图像，便于观察训练初期的小目标检测效果
                    show=False,                # 不弹窗，仅保存图像（服务器推荐）
                    ),)

# 可视化器配置
visualizer = dict(
    type='CustomDetVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='mmengine.TensorboardVisBackend')],
    name='visualizer',
    line_width=0.4,  # 将边界框线条宽度设置为 1 像素
    font_size=5,
    bbox_color=[255., 0., 0.],
    # score_thr=0.1,
)

# 评估器
val_evaluator = dict(
    type='CocoTinyMetric',
    metric=['bbox'],  # 评估检测
    ann_file=data_root + r'annotations\val.json',
    format_only=False,      # =True 只保存预测结果如box cls score等，不做评估
    classwise=False,  # 显示每个类别的AP
    area_Rng= [
        [1, 6.5],      # 2 * 3
        [6.5, 12.5],   # 3 * 4
        [12.5, 24.5],  # 4 * 6
        [24.5, 32.5],  # 4 * 8
        [32.5, 50.5],  # 5 * 10
        [50.5, 336.5], # 14 * 24
    ],
    area_RngLbl = ['<=2*3',
                    '<=3*4',
                    '<=4*6',
                    '<=4*8',
                    '<=5*10',
                    '<=14*25',
                    ]
)