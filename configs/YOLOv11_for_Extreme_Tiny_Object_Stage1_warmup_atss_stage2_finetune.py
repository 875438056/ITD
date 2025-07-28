# YOLOv11 Configuration for Tiny Object Detection
# Optimized for 3x3 pixel objects with high-resolution feature maps

# 关键步骤：告诉 MMDetection 导入你的自定义模块
# MMDetection 在运行时会自动执行这个 import 操作，从而将你的 lightweight_hrnet 注册到 MODELS 注册表中
custom_imports = dict(
    imports=['models.backbones.YOLOV11_backbone_for_extreme_tiny_object',
             'models.necks.YOLOv11_NECK_for_Extreme_Tiny_Object',
             'models.necks.YOLOV11_Light_NECK_for_Extreme_Tiny_Object',
             'models.necks.YOLOv11_pafpn_p1p5',
             'models.heads.YOLOv11_HEAD_for_Extreme_Tiny_Object',
             'models.visualizers.custom_visualizer',
             'models.transforms.centroid_transforms',
             'models.activations.Mish',
             'models.transforms.custom_packer',
             'models.metrics.CocoMetricTinySize',
             'engine.hooks.AssignerSwitchHook',], allow_failed_imports=False)

# 数据配置 - 针对极小目标优化
data_root = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/'
class_name = ('target',) # 你的类别名称
num_classes = len(class_name)
# 数据集特定配置
dataset_type = 'CocoDataset'

img_scale = (256, 256)  # 使用更高的输入分辨率
batch_size = 32
num_workers = 4

switch_at_epoch = 41 # 切换atss_assigner -->


# Model configuration
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[38.6658, 38.6658, 38.6658],
        std=[2.7087, 2.7087, 2.7087],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='YOLOv11BackboneEnhance',
        arch='s',  # 's', 'm', 'l', 'x' based on model size
        out_indices=(1, 2, 3, 4, 5),  # stride=2,4,8,16,32 for small object detection
        frozen_stages=-1,
        norm_eval=False
    ),
    neck=dict(
        type='YOLOv11PAFPN_P5',
        # 根据YOLOv11Backbone 's'架构的实际输出通道数
        # 这些通道数需要根据具体的backbone实现来确定
        in_channels=[32, 64, 64, 128, 256],  # stride 2,4,8,16,32对应的通道数
        # in_channels=[32, 64, 128, 256, 512],  # stride 2,4,8,16,32对应的通道数
        out_channels=[32, 64, 64, 128, 256],
        # num_csp_blocks=3,
        # use_depthwise=False,
        # use_c3k2=True,  # 启用C3k2块
        # upsample_cfg=dict(scale_factor=2, mode='nearest'),
        # norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        # act_cfg=dict(type='SiLU', inplace=True),
        # init_cfg=None
    ),
    bbox_head=dict(
        type='YOLOv11Head',
        num_classes=num_classes,
        in_channels=[32, 64, 64, 128, 256],  # 5个检测层，对应neck的输出
        featmap_strides=[2, 4, 8, 16, 32],  # 包含高分辨率特征图
        # num_base_priors=1,
        reg_max=16,  # DFL regression range
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),

        # Loss configuration for tiny objects
        # loss_cls=dict(
        #     type='mmdet.CrossEntropyLoss',
        #     use_sigmoid=True,
        #     reduction='none',
        #     loss_weight=1.0  # 增加分类损失权重
        # ),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='log',  # Complete IoU for better small object localization
            eps=1e-6,
            reduction='mean',
            loss_weight=7.5  # 增加回归损失权重
        ),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.375  # DFL损失权重
        ),
        assigner=dict(
            type='ATSSAssigner',
            # num_classes=num_classes,
            # use_ciou=True,
            topk=10,
            alpha=1.0,
            # beta=6.0,
            # eps=1e-9,
            # debug=True # 添加调试模式
        )
    ),
    # 针对极小目标的训练配置
    train_cfg=dict(
        assigner=dict(
            type='ATSSAssigner',
            # num_classes=num_classes,
            # use_ciou=True,
            topk=10,
            alpha=1.0,
            # beta=6.0,
            # eps=1e-9,
            # debug=True # 添加调试模式
        )
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=10000,  # 增加NMS前保留的框数量
        score_thr=0.01,  # 降低分数阈值以检测更多小目标
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=100  # 增加每张图像最大检测数量
    )
)





pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True), ]

train_pipeline = [
    # 针对小目标的数据增强
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=0.00001,
        prob=0.5  # 降低Mosaic概率以保护小目标
    ),
    dict(
        type='RandomAffine',
        max_rotate_degree=10.0,  # 减少旋转角度
        max_translate_ratio=0.1,  # 减少平移范围
        scaling_ratio_range=(0.8, 1.2),  # 保守的缩放范围
        max_shear_degree=2.0,  # 减少剪切变换
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(0.00001, 0.00001, 0.00001)
    ),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')
    )
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='Resize', scale=img_scale),
    # dict(
    #     type='LetterResize',
    #     scale=img_scale,
    #     allow_scale_up=False,
    #     pad_val=dict(img=114)
    # ),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')
    )
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/train.json',
            data_prefix=dict(img='images/'),
            metainfo=dict(
                classes=class_name,
                palette='random'  # <-- 将 palette 定义在 metainfo 内部
            ),
            filter_cfg=dict(filter_empty_gt=False, min_size=0.0001),
            pipeline=pre_transform,

    ),
    pipeline=train_pipeline,
))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(
            classes=class_name,
            palette='random'  # <-- 将 palette 定义在 metainfo 内部
        ),
        test_mode=True,
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

# 评估配置
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

test_evaluator = val_evaluator


default_scope = 'mmdet' # 默认的代码库作用域
# 优化器配置 - 针对小目标检测优化
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,  # 基础学习率
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),  # 全局梯度裁剪
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000  # warmup步数
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=140,  # 训练epoch数
        eta_min=0.0001,
        begin=0,
        end=140,  # 总迭代数
        by_epoch=True,
        convert_to_iter_based=False
    )
]

# 训练配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',
                       draw=True,  # 启用绘图
                       interval=50,  # 每50次iter保存一次图像，便于观察训练初期的小目标检测效果
                       show=False,  # 不弹窗，仅保存图像（服务器推荐）
                       ),
)

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='mmengine.TensorboardVisBackend')
]

visualizer = dict(
    type='CustomDetVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    line_width = 0.2,  # 将边界框线条宽度设置为 1 像素
    font_size = 0.5,
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=120,  # 在训练后期关闭强数据增强
        switch_pipeline=val_pipeline
    ),
    dict(
        type='AssignerSwitchHook', # 对应我们注册的类名
        switch_epoch=switch_at_epoch, # 在第41个epoch开始时切换
        # 定义要切换到的“目标”分配器
        switch_assigner=dict(
            type='TaskAlignedAssigner', # TaskAlignedAssigner 或 BatchTaskAlignedAssigner
            topk=10,
            alpha=1.0,
            beta=6.0
            # ... 您 TaskAlignedAssigner 的所有参数 ...
        )
        )
]

# 运行时配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=140,
    val_interval=1,
    # dynamic_intervals=[(280, 1)]  # 最后20个epoch每个epoch验证一次
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 自动缩放学习率
auto_scale_lr = dict(enable=True, base_batch_size=64)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# Test Time Augmentation配置（推理时使用）
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=3000)
)

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            # [
            #     dict(type='KeepRatioResize', scale=img_scale)
            # ],
            [
                dict(type='mmdet.RandomFlip', prob=1.),
                dict(type='mmdet.RandomFlip', prob=0.)
            ],
            # [
            #     dict(
            #         type='LetterResize',
            #         scale=img_scale,
            #         allow_scale_up=False,
            #         pad_val=dict(img=114)
            #     )
            # ],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                              'scale_factor', 'pad_param', 'flip', 'flip_direction')
                )
            ]
        ]
    )
]

# 针对3x3像素极小目标的额外优化建议：
# 1. 使用更高的输入分辨率 (1280x1280 或更高)
# 2. 增加高分辨率特征层的权重
# 3. 使用更保守的数据增强策略
# 4. 降低NMS阈值以保留更多候选框
# 5. 增加训练epoch数和更细致的学习率调度
# 6. 考虑使用FocalLoss替代CrossEntropyLoss
# 7. 启用Test Time Augmentation (TTA)

# 重要提醒：
# 1. in_channels需要根据您的YOLOv11Backbone实际输出通道数进行调整
# 2. 如果backbone输出通道数不匹配，请相应修改neck的in_channels参数
# 3. 确保custom_imports中的模块路径正确，指向您实际的neck实现位置