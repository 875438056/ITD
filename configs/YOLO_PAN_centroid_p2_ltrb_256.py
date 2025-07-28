# ------------------------------------------------------------------
# my_litehrnet_model_config.py
# ------------------------------------------------------------------

# 1. 自定义导入LiteHRNet
# ------------------------------------------------------------------
# 关键步骤：告诉 MMDetection 导入你的自定义模块
# MMDetection 在运行时会自动执行这个 import 操作，从而将你的 lightweight_hrnet 注册到 MODELS 注册表中
custom_imports = dict(
    imports=['models.backbones.YOLO_SPPF',
             'models.necks.PAN_neck_p2',
             'models.visualizers.custom_visualizer',
             'models.transforms.centroid_transforms',
             'models.activations.Mish',
             'models.heads.centroidnet_heatmap_ltrb_offset_head',
             'models.transforms.custom_packer',
             'models.metrics.CocoMetricPerSize'], allow_failed_imports=False)


# 1. 模型定义 (Model Definition)
# ------------------------------------------------------------------
model = dict(
    type='CenterNet',  # 模型类型
    # MMDetection 3.x 新增的数据预处理器，负责归一化、通道转换、打包成批次
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
        std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
        bgr_to_rgb=True),
    # 主干网络：使用 ResNet-18，并启用可变形卷积 (DCN)
    backbone=dict(
        type='YOLOv5sBackbone',  # <-- 在这里直接使用你的自定义 backbone 名称
        depth_multiple=0.33,
        width_multiple=0.50,
        out_indices=(2, 4, 6, 9),
        frozen_stages=-1,
        norm_eval=False,
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=0,
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu'
        ),
        # norm_cfg=dict(type='BN'),
    ),

    # 颈部网络：用于上采样，生成高分辨率特征图
    # <-- 使用自定义的fuse Neck 输出 /stride=2 特征图
    neck=dict(
        type='YOLOv5Neck256WithP1',
        in_channels=[64, 128, 256, 512],
        out_channels=[64, 128, 256, 512, 1024],  # 5个尺度输出
        num_csp_blocks=1,
        upsample_cfg=dict(mode='nearest'),  # 修复：移除scale_factor
    ),

    # 检测头：CenterNet 的核心
    bbox_head=dict(
        type='LTRBCentroidNetHead',
        num_classes=1,  # COCO 数据集有 80 个类别
        in_channels=64,  # neck 的输出通道
        feat_channels=64,
        radius_min=0,   #最小高斯半径
        # 定义三个分支的损失函数
        loss_center_heatmap=dict(
            type='GaussianFocalLoss', loss_weight=1.0),
        loss_ltrb=dict(type='L1Loss', loss_weight=0.5),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    # 训练和测试的特定配置
    train_cfg=None,
    test_cfg=dict(
        topk=100,           # 每张图最多检测 100 个目标
        local_maximum_kernel=5,
        max_per_img=100)    # NMS 后的最终目标数量
)


# 2. 数据集和数据流水线定义 (Dataset and Pipeline)
# ------------------------------------------------------------------
# dataset_type = 'CocoDataset'
dataset_type = 'MultiImageMixDataset'
data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/'
class_name = ('target',) # 你的类别名称
num_classes = len(class_name)

# 训练数据流水线
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),]

#  初始增强 pipeline（含 Mosaic 和 MixUp）
train_pipeline_stage1 = [
    dict(type='Mosaic', img_scale=(256, 256), pad_val=0.00001),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomCenterCropPad',
        crop_size=(256, 256),
        ratios=(1.0, 1.1, 1.2),
        mean=[38.6658, 38.6658, 38.6658],
        std=[2.7087, 2.7087, 2.7087],
        to_rgb=False,
        test_pad_mode=None
    ),
    dict(type='CalculateCentroids'),
    dict(type='PackCustomDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'border'))
]
#  后期增强 pipeline（关闭 Mosaic / MixUp，只保留裁剪等）
train_pipeline_stage2 = [
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomCenterCropPad',
        crop_size=(256, 256),
        ratios=(1.0, 1.1, 1.2),
        mean=[38.6658, 38.6658, 38.6658],
        std=[2.7087, 2.7087, 2.7087],
        to_rgb=False,
        test_pad_mode=None
    ),
    dict(type='CalculateCentroids'),
    dict(type='PackCustomDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'border'))
]


# 测试数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    # dict(type='Resize', scale=(256, 256), keep_ratio=True),
    # dict(type='Pad', size=(256, 256), pad_val=dict(img=(0.00001, 0.00001, 0.00001))),
    dict(type='LoadAnnotations', with_bbox=True),
    # 关键：同样使用 RandomCenterCropPad，但在测试模式下，它会进行确定性的中心填充
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
    dict(
        type='PackCustomDetInputs',
        # 关键：在 meta_keys 中加入 'border'，确保它被打包送入模型
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'border'))
]

# 数据加载器定义 MultiImageMixDataset里嵌套CocoDataset,确保能一次load 2张图片 保证mixup运行

train_dataloader=dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), # 默认的采样器，同时支持分布式和非分布式训练
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline_stage1,
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            ann_file='annotations/train.json',
            data_prefix=dict(img='images/'),
            metainfo=dict(classes=class_name),
            pipeline=pre_transform,
            filter_cfg=dict(filter_empty_gt=True, min_size=0.00001)  # 图片和标注的过滤配置
        ),
    ))

# 修正后的 val_dataloader
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=class_name, palette='random'),
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'), # 图片路径前缀
        test_mode=True, # 开启测试模式，避免数据集过滤图片和标注
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
    optimizer=dict(type='Adam', lr=0.001),
    clip_grad=dict(max_norm=35, norm_type=2)) # 梯度裁剪的配置

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
    checkpoint=dict(type='CheckpointHook', interval=10,
                    save_best='auto'), # 每10个epoch保存一次权重
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',
                       draw=True,  # 启用绘图
                       interval=50,  # 每50次iter保存一次图像，便于观察训练初期的小目标检测效果
                       show=False,  # 不弹窗，仅保存图像（服务器推荐）
                       ), )

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=80,  # 第100轮切换
        switch_pipeline=train_pipeline_stage2
    )
]

# 可视化器配置
visualizer = dict(
    type='CustomDetVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='mmengine.TensorboardVisBackend')],
    name='visualizer',
    line_width=0.2,  # 将边界框线条宽度设置为 1 像素
    font_size=1
)

log_processor = dict(
    type='LogProcessor',  # 日志处理器用于处理运行时日志
    window_size=50,  # 日志数值的平滑窗口
    by_epoch=True)  # 是否使用 epoch 格式的日志。需要与训练循环的类型保存一致。

# 评估器
val_evaluator = dict(
    # type='CocoMetricCustomRanges',
    type='CocoMetric',
    ann_file=data_root + r'annotations\val.json',
    metric=['bbox'],
    # metric=['proposal']
    # format_only=False
    # area_ranges={
    #     '1-6': [1, 6],
    #     '6-12': [6, 12],
    #     '14-24': [14, 24],
    #     '24-32': [24, 32],
    #     '32-50': [32, 50],
    #     '50-200': [50, 200],
    #     '200-450': [200, 450],
    # }
)

test_evaluator = val_evaluator

# 其他运行时设置
log_level = 'INFO'
load_from = None  # 不从预训练权重恢复，因为backbone已指定
resume = False    # 不从断点恢复

