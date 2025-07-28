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
             'models.heads.centroidnet_DSNT_heatmap_ltrb_head',
             'models.transforms.custom_packer',
             'engine.hooks.loss_weight_scheduler_hook',], allow_failed_imports=False)


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
        out_indices=(2, 4, 6, 9),  # 输出特征层索引
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
        upsample_cfg=dict(mode='nearest'),
    ),

    # 检测头：CenterNet 的核心
    bbox_head=dict(
        type='LTRBCentroidNetDSNTHead',
        num_classes=1,  # COCO 数据集有 80 个类别
        in_channels=64,  # neck 的输出通道
        feat_channels=64,
        radius_min=0,   #最小高斯半径
        patch_size=5,
        # 定义三个分支的损失函数
        loss_center_heatmap=dict(
            type='GaussianFocalLoss', loss_weight=1.0),
        loss_ltrb=dict(type='L1Loss', loss_weight=0.5),
        loss_centroid=dict(type='L1Loss', loss_weight=0.5)),
    # 训练和测试的特定配置
    train_cfg=None,
    test_cfg=dict(
        topk=100,           # 每张图最多检测 100 个目标
        local_maximum_kernel=3,
        max_per_img=50)    # NMS 后的最终目标数量
)


# 2. 数据集和数据流水线定义 (Dataset and Pipeline)
# ------------------------------------------------------------------
dataset_type = 'MultiImageMixDataset'
data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/'
class_name = ('target',) # 你的类别名称
num_classes = len(class_name)

# 训练数据流水线
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#
#     # CenterNet 常用数据增强
#     dict(type='RandomFlip', prob=0.5),
#     # 随机裁剪，这是 CenterNet 训练中很重要的增强
#     dict(
#         type='RandomCenterCropPad',
#         crop_size=(256, 256),
#         ratios=(1.0, 1.1, 1.2, 1.3),
#         mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
#         std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
#         to_rgb=True,
#         test_pad_mode=None),
#     # dict(type='Resize', scale=(256, 256), keep_ratio=True),
#     # 将标注信息转换为 CenterNet 的训练目标（热图、宽高、偏移量）
#     # dict(type='GenerateCenterNetTargets'),
#     dict(type='CalculateCentroids'),  # <-- 在这里添加计算质心的新模块
#     dict(type='PackCustomDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                     'scale_factor', 'border'),
#         # pack_keys=('gt_bboxes', 'gt_labels', 'gt_centroids') # 打包成 MMDetection 需要的输入格式
# )]

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
    # 同样先进行一次缩放，确保最长边不超过目标尺寸
    # dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(
        type='PackCustomDetInputs',
        # 关键：在 meta_keys 中加入 'border'，确保它被打包送入模型
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'border'))
]

# 数据加载器定义
train_dataloader=dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
            filter_cfg=dict(filter_empty_gt=True, min_size=0.00001)
        ),
    ))
# 修正后的 val_dataloader
val_dataloader = dict(
    batch_size=24,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
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
    # optimizer=dict(type='SGD', lr=0.01, momentum=0.9, nesterov=True),
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05), # lr一般范围：1e-5 到 1e-3, 较小数据集/模型可以使用较大学习率(如1e-3),反之亦然
    clip_grad=dict(max_norm=35, norm_type=2))  # 梯度裁剪通过对梯度进行缩放，使其不超过设定的最大范数
                                               # 2：L2范数（默认，计算所有梯度元素的平方和再开方）。1：L1范数（绝对值之和）

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR',       # 作用：学习率从lr * start_factor线性增长到lr * end_factor
         start_factor=0.001,    # 初始学习率 = lr * start_factor
         end_factor=1,
         by_epoch=False,       # 最终学习率 = lr * end_factor（默认1.0
         begin=0,
         end=500),    # 更短warmup
    # dict( # 常数预热
    #     type='ConstantLR',
    #     factor=0.001,       # 学习率 = lr * factor
    #     by_epoch=False,
    #     begin=0,
    #     end=500
    # ),
    # dict( # 余弦退火
    #     type='CosineAnnealingLR',
    #     T_max=100,  # 假设总epoch=100
    #     eta_min=1e-6,
    #     by_epoch=True,
    #     begin=0,  # 从第0 epoch开始
    #     end=100
    # ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        by_epoch=True,
        milestones=[90, 120],  # 在第90和120个epoch降低学习率
        gamma=0.1)   # 衰减系数 每次衰减为之前的0.1倍
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
                        draw=True,  # 确保开启绘制
                        interval=50,  # 每100个iter保存一次
                        show=False),  # 不在屏幕上显示，只保存到文件
)

custom_hooks = [
    dict(type='PipelineSwitchHook', switch_epoch=50, switch_pipeline=train_pipeline_stage2 ), # 第100轮切换
    dict(
        type='LossWeightSchedulerHook',
        loss_name='loss_centroid',  # 对应 LTRBCentroidNetDSNTHead 中的 loss_centroid
        start_epoch=50,  # 从第50个epoch结束后开始调整
        end_epoch=100,  # 到第100个epoch结束时达到最终权重
        start_weight=0.05,  # 初始权重
        end_weight=1  # 最终权重
    )
]

# 可视化器配置
visualizer = dict(
    type='CustomDetVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='mmengine.TensorboardVisBackend')],
    name='visualizer',
    line_width=0.2,  # 将边界框线条宽度设置为 1 像素
    font_size=0.5
)

# 评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + r'annotations\val.json',
    metric=['bbox']
    # metric=['proposal']
    # format_only=False
)

test_evaluator = val_evaluator

# 其他运行时设置
log_level = 'INFO'
load_from = None  # 不从预训练权重恢复，因为backbone已指定
resume = False    # 不从断点恢复

