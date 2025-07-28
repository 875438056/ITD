# ------------------------------------------------------------------
# my_litehrnet_model_config.py
# ------------------------------------------------------------------

# 1. 自定义导入LiteHRNet
# ------------------------------------------------------------------
# 关键步骤：告诉 MMDetection 导入你的自定义模块
# MMDetection 在运行时会自动执行这个 import 操作，从而将你的 lightweight_hrnet 注册到 MODELS 注册表中
custom_imports = dict(
    imports=['models.backbones.lightweight_hrnet',
             'models.backbones.lightweight_hrnet_p2',
             'models.necks.identity_neck',
             'models.visualizers.custom_visualizer',
             'models.activations.Mish',
             'models.metrics.CocoMetricTinySize'], allow_failed_imports=False)

# 2. 定义 LiteHRNet 需要的 extra 配置
# ------------------------------------------------------------------
litehrnet_extra = dict(
    stem=dict(
        stem_channels=32,
        out_channels=32,
        expand_ratio=1),
    num_stages=3,
    stages_spec=dict(
        num_modules=(2, 4, 2),
        num_branches=(2, 3, 4),
        num_blocks=(2, 2, 2),
        module_type=('LITE', 'LITE', 'LITE'),
        with_fuse=(True, True, True),
        reduce_ratios=(8, 8, 8),
        num_channels=(
            (40, 80),
            (40, 80, 160),
            (40, 80, 160, 320),
        )),
    with_head=True  # 在 MMDetection 中作为 backbone 使用时，必须为 False
)


# 1. 模型定义 (Model Definition)
# ------------------------------------------------------------------
model = dict(
    type='CenterNet',  # 模型类型
    # MMDetection 3.x 新增的数据预处理器，负责归一化、通道转换、打包成批次
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[38.6658, 38.6658, 38.6658],
        std=[2.7087, 2.7087, 2.7087],
        bgr_to_rgb=True),
    # 主干网络：使用 ResNet-18，并启用可变形卷积 (DCN)
    # backbone=dict(
    #     type='ResNet',
    #     depth=18,
    #     norm_eval=False,
    #     norm_cfg=dict(type='BN'),
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    # 主干网络：使用 ResNet-18，并启用可变形卷积 (DCN)
    backbone=dict(
        type='lightweight_hrnet_p2',  # <-- 在这里直接使用你的自定义 backbone 名称
        extra=litehrnet_extra,
        norm_cfg=dict(type='BN')
    ),

    # <-- 使用自定义的直通 Neck
    neck=dict(
        type='SelectFeatureNeck',
        in_index=0),    # Neck 选择第一个（最高分辨率）的特征图

    # 检测头：CenterNet 的核心
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,  # COCO 数据集有 80 个类别
        in_channels=40,  # neck 的输出通道
        feat_channels=40,
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
        max_per_img=3),    # NMS 后的最终目标数量
    init_cfg=dict(
        type='Pretrained',
        # checkpoint=r"F:\ITD\tools\work_dirs\lw_hrnet_centernet_p2_wh_256\20250722_004627\best_coco_target_precision_epoch_137.pth",
        checkpoint=r"F:\ITD\tools\work_dirs\lw_hrnet_centernet_p2_wh_256\20250723_095535\best_coco_bbox_mAP_epoch_293.pth",
    )
)


# 2. 数据集和数据流水线定义 (Dataset and Pipeline)
# ------------------------------------------------------------------
dataset_type = 'CocoDataset'
# data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/'
# data_root = r'E:\DATASET\ITT\bbox_format\IRSTD-1k/'
data_root = r'E:\DATASET\ITT\bbox_format\SIRST/'
class_name = ('target',) # 你的类别名称
num_classes = len(class_name)

# 训练数据流水线
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#
#     # CenterNet 常用数据增强
#     # dict(
#     #     type='PhotoMetricDistortion',
#     #     brightness_delta=32,
#     #     contrast_range=(0.5, 1.5),
#     #     saturation_range=(0.5, 1.5),
#     #     hue_delta=18),
#     # 随机裁剪，这是 CenterNet 训练中很重要的增强
#     # dict(
#     #     type='RandomCenterCropPad',
#     #     crop_size=(256, 256),
#     #     ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
#     #     mean=[38.6658, 38.6658, 38.6658],
#     #     std=[2.7087, 2.7087, 2.7087],
#     #     to_rgb=True,
#     #     test_pad_mode=None),
#     # dict(type='Resize', scale=(256, 256), keep_ratio=True),
#     # dict(type='RandomFlip', prob=0.5),
#     # 将标注信息转换为 CenterNet 的训练目标（热图、宽高、偏移量）
#     # dict(type='GenerateCenterNetTargets'),
#     # 随机裁剪，这是 CenterNet 训练中很重要的增强
#     dict(
#         type='RandomCenterCropPad',
#         crop_size=None,
#         # ratios=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
#         ratios=None,
#         border=None,
#         mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
#         std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
#         to_rgb=True,
#         test_mode=True,
#         test_pad_mode=['size_divisor', 32], ),
#     dict(type='PackDetInputs') # 打包成 MMDetection 需要的输入格式
# ]

# test_pipeline = [
#             dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
#             dict(type='Resize', scale=(256, 256), keep_ratio=True),
#             dict(
#                 type='RandomCenterCropPad',
#                 crop_size=None,
#                 ratios=None,
#                 border=None,
#                 # 补上缺失的参数
#                 mean=[38.6658, 38.6658, 38.6658],
#                 std=[2.7087, 2.7087, 2.7087],
#                 to_rgb=True,
#                 test_mode=True,
#                 test_pad_mode=['size_divisor', 32],
#             ),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 type='PackDetInputs',
#                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                            'scale_factor', 'border'))
#         ]

# 训练数据流水线
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),

    # CenterNet 常用数据增强
    # dict(type='Resize', scale=(256, 256), keep_ratio=True),
    # 随机裁剪，这是 CenterNet 训练中很重要的增强
    dict(  # 不使用RandomCenterCropPad，但是传入border参数（后续pre需要）
        type='RandomCenterCropPad',
        crop_size=(256, 256),
        # # ratios=(1.1, 1.2, 1.3),
        ratios=(1.0,),
        # crop_size=None,  # 在测试时，我们不进行随机裁剪
        # ratios=None,
        # border=None,
        mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
        std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
        to_rgb=True,
        test_mode=False,
        # test_pad_mode=['size_divisor', 32],
        test_pad_mode=None,
    ),

    # dict(type='RandomFlip', prob=0.5),
    # 将标注信息转换为 CenterNet 的训练目标（热图、宽高、偏移量）
    # dict(type='GenerateCenterNetTargets'),
    # dict(type='CalculateCentroids'),  # <-- 在这里添加计算质心的新模块
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'border'),
        # pack_keys=('gt_bboxes', 'gt_labels', 'gt_centroids') # 打包成 MMDetection 需要的输入格式
)]

# 测试数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # 同样先进行一次缩放，确保最长边不超过目标尺寸
    # dict(type='Resize', scale=(256, 256), keep_ratio=True),
    # 关键：同样使用 RandomCenterCropPad，但在测试模式下，它会进行确定性的中心填充
    dict(
        type='RandomCenterCropPad',
        # crop_size=None,  # 在测试时，我们不进行随机裁剪
        # crop_size=(256, 256),
        # ratios=(1.0,),
        ratios=None,
        border=None,
        # 补上缺失的参数
        mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
        std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
        to_rgb=True,
        test_mode=True,  # 开启测试模式
        # test_pad_mode=['logical_or', 31],  # 这是 CenterNet 官方测试时常用的 padding 模式
        test_pad_mode=['size_divisor', 32],
        # test_pad_mode=None,
    ),

    dict(
        type='PackDetInputs',
        # 关键：在 meta_keys 中加入 'border'，确保它被打包送入模型
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'border'))
]

# 数据加载器定义
train_dataloader = dict(
    batch_size=12,
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
    batch_size=12,
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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器封装
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.00001),
    clip_grad=dict(max_norm=35, norm_type=2))

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        by_epoch=True,
        milestones=[90, 180],  # 在第90和120个epoch降低学习率
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
                        draw=True,  # 确保开启绘制
                        interval=10,  # 每100个iter保存一次
                        show=False),  # 不在屏幕上显示，只保存到文件

)

# 可视化器配置
visualizer = dict(
    type='CustomDetVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='mmengine.TensorboardVisBackend')],
    name='visualizer',
    line_width=0.4,  # 将边界框线条宽度设置为 1 像素
    font_size=5,
    # score_thr=0.6,
    # alpha=0.4,
)

# 评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + r'annotations\val.json',
    metric='bbox',
    format_only=False,
    # classwise = True,  # 如果需要按类别计算
)
# 评估器
# val_evaluator = dict(
#     type='CocoTinyMetric',
#     metric=['bbox'],  # 评估检测
#     ann_file=data_root + r'annotations\val.json',
#     format_only=False,      # =True 只保存预测结果如box cls score等，不做评估
#     classwise=False,  # 显示每个类别的AP
#     area_Rng= [
#         [1, 6.5],      # 2 * 3
#         [6.5, 12.5],   # 3 * 4
#         [12.5, 24.5],  # 4 * 6
#         [24.5, 32.5],  # 4 * 8
#         [32.5, 50.5],  # 5 * 10
#         [50.5, 336.5], # 14 * 24
#     ],
#     area_RngLbl = ['<=2*3',
#                     '<=3*4',
#                     '<=4*6',
#                     '<=4*8',
#                     '<=5*10',
#                     '<=14*25',
#                     ]
# )

test_evaluator = val_evaluator

# 其他运行时设置
log_level = 'INFO'
load_from = None  # 不从预训练权重恢复，因为backbone已指定
resume = False    # 不从断点恢复