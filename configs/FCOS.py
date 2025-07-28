# 继承自官方的 FCOS 基础配置

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
    r'F:\ITD\mmdetection\configs\fcos\fcos_r50-caffe_fpn_gn-head_1x_coco.py'
]

# --- 1. 修改模型 ---
# model = dict(
#     bbox_head=dict(
#         num_classes=1  # !!! 重要：修改为你的数据集的类别数量
#     )
# )
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
        std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        base_channels=32,  # 原通道：[256, 512, 1024, 2048] （对应 base_channels=64） 新通道：[128, 256, 512, 1024] （减半后）
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # 输出所有4个阶段的特征
        strides=(1, 2, 2, 2),  # 每个阶段的第一个block的stride
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='open-mmlab://detectron/resnet50_caffe')
        init_cfg=None),
    neck=dict(
        type='FPN',
        # in_channels=[256, 512, 1024, 2048],
        # in_channels=[256, 512, 1024],
        in_channels=[128, 256, 512, 1024],
        out_channels=128,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=4,
        end_level=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=128,
        stacked_convs=4,
        feat_channels=128,
        # strides=[8, 16, 32, 64, 128],
        strides=[4, 8, 16, 32],
        regress_ranges=(
            (-1, 16),    # 4x下采样层（预测微小/小目标）
            (16, 64),   # 8x下采样层（预测小/中目标）
            (64, 128),  # 16x下采样层（预测中/大目标）
            (128, 100000)   # 32x下采样层（预测大/巨大目标）
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=100))
# --- 2. 修改数据集和数据加载器 ---
# 数据集类型
dataset_type = 'CocoDataset'
# 数据集根目录
data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/'

# 定义数据集的类别名称
metainfo = {
    'classes': ('target'),
    # 'palette': [(220, 20, 60), (119, 11, 32), ...] # (可选) 为每个类别定义颜色
}

# 修改训练数据加载器
train_dataloader = dict(
    batch_size=16,  # 根据你的 GPU 显存调整
    num_workers=4, # 根据你的 CPU 核心数调整
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',  # 训练标注文件路径
        data_prefix=dict(img='images/')      # 训练图片路径前缀
    )
)

# 修改验证数据加载器
val_dataloader = dict(
    batch_size=16,  # 根据你的 GPU 显存调整
    num_workers=4,  # 根据你的 CPU 核心数调整
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',    # 验证标注文件路径
        data_prefix=dict(img='images/')        # 验证图片路径前缀
    )
)

# 修改测试数据加载器
test_dataloader = val_dataloader

# --- 3. 修改评估方式 ---
val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json' # 验证时使用的标注文件
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


# --- 4. 修改训练计划和运行设置 ---
# 加载预训练模型可以显著加快收敛速度并提高性能
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-d9e3d4ce.pth'

# 设置工作目录，用于保存日志和模型权重
# work_dir = './work_dirs/my_fcos_r50_fpn'

# (可选) 调整学习率
# optim_wrapper = dict(
#     optimizer=dict(lr=0.005) # 默认是 0.01，如果 batch_size 减半，学习率也建议减半
# )