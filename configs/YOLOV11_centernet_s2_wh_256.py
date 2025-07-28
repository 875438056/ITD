# ------------------------------------------------------------------
# my_litehrnet_model_config.py
# ------------------------------------------------------------------

# 1. 自定义导入LiteHRNet
# ------------------------------------------------------------------
# 关键步骤：告诉 MMDetection 导入你的自定义模块
# MMDetection 在运行时会自动执行这个 import 操作，从而将你的 lightweight_hrnet 注册到 MODELS 注册表中
custom_imports = dict(
    imports=['models.backbones.lightweight_hrnet',
             'models.backbones.yolo11_Backbone',
             'models.necks.yolo11_pafpn',
             'models.necks.identity_neck',
             'models.visualizers.custom_visualizer',
             'models.metrics.CocoMetricTinySize',], allow_failed_imports=False)


# 1. 模型定义 (Model Definition)
# ------------------------------------------------------------------
deepen_factor = 0.33
widen_factor = 0.50

# 根据 widen_factor 计算的通道数
in_channels = [32, 64, 128, 256]

model = dict(
    type='CenterNet',  # 模型类型
    # MMDetection 3.x 新增的数据预处理器，负责归一化、通道转换、打包成批次
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),

    backbone=dict(
        type='YOLOv11Backbone',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=3,
        out_indices=(0, 2, 4, 6),  # P1, P2, P3, P4对应的层索引
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=0,
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu'
        ),
    ),
    # 颈部网络：用于上采样，生成高分辨率特征图
    neck=dict(
        type='YOLOv11PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=in_channels,
        out_channels=in_channels,
    ),

    # 检测头：CenterNet 的核心
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,  # COCO 数据集有 80 个类别
        in_channels=64,  # neck 的输出通道
        feat_channels=64,
        # 定义三个分支的损失函数
        loss_center_heatmap=dict(
            type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    # 训练和测试的特定配置
    train_cfg=None,
    test_cfg=dict(
        topk=100,           # 每张图最多检测 100 个目标
        local_maximum_kernel=3,
        max_per_img=3)    # NMS 后的最终目标数量
)


# 2. 数据集和数据流水线定义 (Dataset and Pipeline)
# ------------------------------------------------------------------
dataset_type = 'CocoDataset'
# data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_2803(14015)/'
data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/'
class_name = ('target',) # 你的类别名称
num_classes = len(class_name)

# 训练数据流水线
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),

    # 数据增强
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转 # 翻转概率50%

    dict(type='RandomAffine',
         max_rotate_degree=2,  # 旋转角度范围（-N° ~ +N°）
         max_translate_ratio=0.1,  # 平移比例（相对于图像尺寸) ratio*width
         scaling_ratio_range=(1.1, 1.2),  # 缩放比例范围
         max_shear_degree=1,  # 剪切角度范围
         border_val=0.0000001),  # 填充边界值（BGR格式）

    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='Normalize', mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
         std=[2.708734413355478, 2.708734413355478, 2.708734413355478]),
    dict(type='PackDetInputs', # 打包成 MMDetection 需要的输入格式
    meta_keys = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                 'scale_factor', 'border')
                  ),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(
        type='RandomCenterCropPad',
        crop_size=None,  # 在测试时，我们不进行随机裁剪
        ratios=None,
        border=None,
        # 补上缺失的参数
        mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
        std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
        to_rgb=True,
        test_mode=True,  # 开启测试模式
        test_pad_mode=['logical_or', 31],  # 这是 CenterNet 官方测试时常用的 padding 模式
    ),

    # dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='Normalize', mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
         std=[2.708734413355478, 2.708734413355478, 2.708734413355478]),
    dict(
        type='PackDetInputs',
        # 关键：在 meta_keys 中加入 'border'，确保它被打包送入模型
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'border')
        )
]

# 数据加载器定义
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(
            classes=class_name,
            palette='random'  # <-- 将 palette 定义在 metainfo 内部
        ),
        # filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline))

# 修正后的 val_dataloader
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=class_name, palette='random'),
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline))

# test_dataloader 通常与 val_dataloader 一致
test_dataloader = val_dataloader

# 3. 训练策略定义 (Training Schedule)
# ------------------------------------------------------------------
# 训练、验证、测试的循环配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=140, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器封装
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

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

# 4. 运行时设置 (Runtime Settings)
# ------------------------------------------------------------------
default_scope = 'mmdet' # 默认的代码库作用域
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='auto'), # 每10个epoch保存一次权重
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',
                       draw=True,  # 启用绘图
                       interval=50,  # 每50次iter保存一次图像，便于观察训练初期的小目标检测效果
                       show=False,  # 不弹窗，仅保存图像（服务器推荐）
                       ), )

# 可视化器配置
visualizer = dict(
    type='CustomDetVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='mmengine.TensorboardVisBackend')],
    name='visualizer',
    line_width=0.3,  # 将边界框线条宽度设置为 1 像素
    font_size=1
)

# 评估器
val_evaluator = dict(
    type='CocoTinyMetric',
    metric=['bbox'],  # 评估检测
    ann_file=data_root + r'annotations\val.json',
    format_only=False,      # =True 只保存预测结果如box cls score等，不做评估
    classwise=False,  # 显示每个类别的AP
    # outfile_prefix='./work_dirs/coco_detection/val'
    # metric_items=[
    #     'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
    #     'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000'
    # ],
    # metric_items=[
    #     'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'AR@100','AR_s@1000',],
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

test_evaluator = val_evaluator

# 其他运行时设置
log_level = 'INFO'
load_from = None  # 不从预训练权重恢复，因为backbone已指定
resume = False    # 不从断点恢复

