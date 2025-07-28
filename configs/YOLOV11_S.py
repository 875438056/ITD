
# 关键步骤：告诉 MMDetection 导入你的自定义模块
# MMDetection 在运行时会自动执行这个 import 操作，从而将你的 lightweight_hrnet 注册到 MODELS 注册表中
custom_imports = dict(
    imports=['models.backbones.yolo11_Backbone',
             'models.necks.yolo11_pafpn',
             'models.visualizers.custom_visualizer',
             'models.transforms.centroid_transforms',
             'models.activations.Mish',
             'models.heads.yolo11_head',
             'models.heads.yolov11_s2_head',
             'models.transforms.custom_packer'], allow_failed_imports=False)


deepen_factor = 0.33
widen_factor = 0.50

# 根据 widen_factor 计算的通道数
in_channels = [32, 64, 128, 256]

# 1. 模型定义 (Model Definition)
# ------------------------------------------------------------------
model = dict(
    type='SingleStageDetector',
    # 数据预处理器，用于归一化等
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[38.66580139016556, 38.66580139016556, 38.66580139016556],
        std=[2.708734413355478, 2.708734413355478, 2.708734413355478],
        bgr_to_rgb=True),

    backbone=dict(
        type='YOLOv11Backbone',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=3,
        out_indices=(0, 2, 4, 6), # P1, P2, P3, P4对应的层索引
        init_cfg=dict(
                    type='Kaiming',
                    layer='Conv2d',
                    a=0,
                    distribution='uniform',
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                ),
    ),

    neck=dict(
        type='YOLOv11PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=in_channels,
        out_channels=in_channels,
    ),

    # bbox_head=dict(
    #     type='YOLOv11Head',
    #     head_module=dict(
    #         type='YOLOv11HeadModule',
    #         num_classes=1, # COCO数据集类别数
    #         in_channels=in_channels,
    #         widen_factor=widen_factor,
    #         featmap_strides=[2, 4, 8, 16]
    #     ),
    #     prior_generator=dict(
    #         type='mmdet.MlvlPointGenerator', offset=0.5, strides=[2, 4, 8, 16]),
    #     bbox_coder=dict(type='DistancePointBBoxCoder'), # 将“从一个点到四条边的距离lrtb”这种表示法与标准的(x1, y1, x2, y2)坐标进行相互转换
    #
    #     # 损失函数定义
    #     loss_cls=dict(
    #         type='mmdet.CrossEntropyLoss',
    #         use_sigmoid=True,
    #         reduction='none',
    #         loss_weight=1.0),
    #     loss_bbox=dict(
    #         type='mmdet.IoULoss',
    #         mode='log',
    #         # reduction='sum',
    #         loss_weight=1.0,),
    #     loss_dfl=dict(
    #         type='mmdet.DistributionFocalLoss',
    #         reduction='mean',
    #         loss_weight=0.5),
    #     test_cfg=dict(
    #         score_thr=0.01,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         max_per_img=300,
    #     ),
    # ),

    bbox_head=dict(
        type='YOLOv11S2Head',
        num_classes=1,
        in_channels=[128, 256, 512, 1024],  # P1, P2, P3, P4对应的通道数
        # 其他参数...
    ),


        # 训练配置
        train_cfg=dict(
            assigner=dict(
                type='mmdet.TaskAlignedAssigner', # 动态标签分配
                topk=10,),  # exp(beta * IoU) 通过指数放大回归质量的影响

        # 测试配置
        test_cfg=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300,
    )
    ))

# 2. 数据集和数据流水线定义 (Dataset and Pipeline)
# ------------------------------------------------------------------
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
        ratios=(1.0,),
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
        ratios=(1.0,),
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
    batch_size=48,
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
    batch_size=48,
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
    optimizer=dict(type='Adam', lr=0.001),
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + r'annotations\val.json',
    metric=['bbox'],
)

test_evaluator = val_evaluator

# 其他运行时设置
log_level = 'INFO'
load_from = None  # 不从预训练权重恢复，因为backbone已指定
resume = False    # 不从断点恢复
