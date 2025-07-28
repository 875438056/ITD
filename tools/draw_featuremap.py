from mmengine.visualization import Visualizer
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmengine.config import Config
import torch
from torchvision import transforms
from mmdet.utils import register_all_modules
import numpy as np
import cv2
import os
from datetime import datetime
register_all_modules()

# r'F:\ITD\configs\lw_hrnet_centernet.py'
# r"F:\ITD\tools\work_dirs\lw_hrnet_centernet\epoch_140.pth"
# "F:\ITD\tools\work_dirs\YOLO_PAN_centroid_p2_ltrb_adaptive_256\epoch_140.pth"
# F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_256
# "F:\ITD\tools\work_dirs\lw_hrnet_centernet_s4_ltrb_256\best_coco_bbox_mAP_epoch_5.pth"
# F:\ITD\tools\work_dirs\lw_hrnet_centernet_p2_ltrb_256\20250626_152427

# r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_256\best_coco_bbox_mAP_epoch_106.pth"
# 1. 读取配置和模型
# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250702_142918\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250703_021259\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250702_142918\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\20250714_015252\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1.py"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\20250714_015252\best_coco_bbox_mAP_epoch_49.pth"
# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1.py"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\best_coco_bbox_mAP_epoch_3.pth"

# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250703_170635\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250703_170635\epoch_60.pth"

# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250703_021259\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250703_021259\best_coco_bbox_mAP_epoch_23.pth"

# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250704_012941\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250704_012941\best_coco_bbox_mAP_epoch_16.pth"

# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\20250701_021454\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1.py"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\20250701_021454\best_coco_bbox_mAP_epoch_38.pth"

# cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250721_004320\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250721_004320\best_coco_bbox_mAP_epoch_66.pth"

cfg_root = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250719_124104\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1.py"
checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250719_124104\best_coco_bbox_mAP_epoch_18.pth"


cfg_file_name = os.path.splitext(os.path.basename(cfg_root))[0]
cfg = Config.fromfile(cfg_root)

model = MODELS.build(cfg.model)
print('*'*15 + '网络模型已载入' + '*'*15)
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250702_142918\best_coco_bbox_mAP_epoch_44.pth"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250703_021259\best_coco_bbox_mAP_epoch_23.pth"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb(iou)_adaptive_256_GaussianFocalLoss_L1\20250702_142918\best_coco_bbox_mAP_epoch_44.pth"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\20250701_160048\best_coco_bbox_mAP_epoch_22.pth"
# checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centroid_p2_ltrb_adaptive_256_GaussianFocalLoss_L1\20250701_110541\best_coco_bbox_mAP_epoch_14.pth"
# checkpoint = r"F:\ITD\tools\work_dirs\YOLO_PAN_centroid_DSNT_p2_ltrb_256\best_coco_bbox_mAP_epoch_66.pth"
load_checkpoint(model, checkpoint, map_location='cpu')
print('*'*15 + '模型权重已载入' + '*'*15)

model.eval()  # 切换到 eval 模式
model.cuda()  # 如果有 GPU

# 2. 准备图片并预处理
from mmengine.dataset import Compose
test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
train_pipeline = Compose(cfg.train_dataloader.dataset.pipeline)

print('*'*15 + '载入图片....' + '*'*15)
# data_path = r"E:\DATASET\LJB\LJB_jpg_test"
# image_name = '4_6_913.jpg'

data_path = r'E:\DATASET\LJB\2_15'
image_name = '2_15_935.jpg'
# data_path = r"E:\DATASET\LJB\LJB_train_coco_jpg_latest_trainset_14015\images"
# image_name = '5_15_1000.jpg'
# image_name = '5_15_750.jpg'
img_path = os.path.join(data_path, image_name)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W] 转换为0-1之间的浮点数
# print(img_tensor.shape, img_tensor.dtype)

# 经过test_pipeline
data = dict(img_path=img_path)
pre_inputs = test_pipeline(data)
pre_inputs['inputs'] = test_pipeline(data)['inputs'].unsqueeze(0)
pre_inputs['data_samples'] = None

# target_inputs = train_pipeline(data)
# target_inputs['inputs'] = train_pipeline(data)['inputs'].unsqueeze(0)
# target_inputs['data_samples'] = None
# inputs_batch = inputs.unsqueeze(0)
# print(inputs_batch.shape, inputs_batch.dtype)
# print('*'*15 + '<UNK>' + '*'*15)

# processed = model.data_preprocessor(inputs, False)


# 构造 data dict 直接输入原图
# inputs = dict(inputs=img_tensor, data_samples=None)
# print(inputs)
# print(inputs.shape, inputs.dtype)
pre_processed = model.data_preprocessor(pre_inputs, False)
print('*'*15 + '数据处理已完成' + '*'*15)

print('*'*15 + '正在读取FM..' + '*'*15)
# 3. 获取中间特征图，比如 backbone 输出
with torch.no_grad():
    pre_inputs = pre_processed['inputs'].cuda()
    # feat_maps = model.backbone(inputs)[2]  # 取第一级特征图 (N, C, H, W)
    feat_maps = model.backbone(pre_inputs)

# 生成heatmap
outs = model.neck(feat_maps)
head_out = model.bbox_head(outs)
heatmap = head_out[0]

# 4. 使用 Visualizer 画出特征图
vis = Visualizer()

pre_vis_feat_maps = vis.draw_featmap(
    heatmap[0].squeeze(0),  # torch.Tensor (C, H, W) or (N, C, H, W)
    img,
    channel_reduction='squeeze_mean',  # 可选 'select_max', 'select_min', 'squeeze_mean'
    resize_shape=(256, 256), # 可调节输出大小
    alpha=0.3,
    topk=3,
)

# vis_feat_maps = vis.draw_featmap(
#     feat_maps.squeeze(0),  # torch.Tensor (C, H, W) or (N, C, H, W)
#     img,
#     channel_reduction='select_max',  # 可选 'select_max', 'select_min', 'squeeze_mean'
#     resize_shape=(256, 256), # 可调节输出大小
#     alpha=0.2,
#     topk=3,
# )

save_dir = r'F:\ITD\vis_featuremap'
formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_file = os.path.join(save_dir, formatted_time+'_'+cfg_file_name+'_featuremap_'+image_name)

cv2.imwrite(save_file, cv2.cvtColor(pre_vis_feat_maps, cv2.COLOR_RGB2BGR))
print(f"可视化特征图已保存到 {save_file}")