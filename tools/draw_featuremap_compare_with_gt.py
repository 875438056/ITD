import torch
import cv2
import os
import numpy as np
from datetime import datetime
import json
from types import SimpleNamespace
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

# MMDetection/MMCV imports
from mmengine.dataset import Compose
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.visualization import Visualizer
from mmdet.registry import MODELS, TRANSFORMS
from mmdet.utils import register_all_modules
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


# 定义一个支持 len() 和 .tensor 属性的简单模拟类，以更好地模拟MMDetection的Box类型
class MockBoxes:
    def __init__(self, tensor_data):
        self.tensor = tensor_data

    def __len__(self):
        return len(self.tensor)


# ======================================================================
# 2. 辅助函数：自动从COCO JSON文件查找BBox (无变化)
# ======================================================================
def find_bboxes_for_image(image_name, json_paths):
    """
    Searches through COCO JSON files to find bounding boxes for a given image name.
    """
    for json_path in json_paths:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        filename_to_id = {img['file_name']: img['id'] for img in data['images']}
        if image_name in filename_to_id:
            image_id = filename_to_id[image_name]
            found_bboxes = []
            for ann in data['annotations']:
                if ann['image_id'] == image_id:
                    x, y, w, h = ann['bbox']
                    found_bboxes.append([x, y, x + w, y + h])
            if found_bboxes:
                return found_bboxes
    return []


# ======================================================================
# 3. 主程序
# ======================================================================

# 注册所有模块
register_all_modules()

# --- 模型和配置加载 ---
cfg_root = r'F:/ITD/configs/lw_hrnet_centernet_p2_wh_256.py'  # 确保配置文件正确
cfg_file_name = os.path.splitext(os.path.basename(cfg_root))[0]
cfg = Config.fromfile(cfg_root)
model = MODELS.build(cfg.model)
checkpoint = r"F:\ITD\tools\work_dirs\lw_hrnet_centernet_p2_wh_256/best_coco_bbox_mAP_epoch_124.pth"
load_checkpoint(model, checkpoint, map_location='cpu')
model.eval()
model.cuda()
print("模型和权重加载完毕。")

# --- 图像和标注加载 ---
test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
image_name = '3_15_500.jpg'
data_root = r"E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015"
img_path = os.path.join(data_root, 'images', image_name)
img = cv2.imread(img_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"图片 '{image_name}' 加载完毕。")

# --- 生成预测 Heatmap ---
data_for_pred = dict(img_path=img_path, img=img.copy(), img_shape=img.shape[:2])
pre_inputs_dict = test_pipeline(data_for_pred)
pre_inputs_tensor = pre_inputs_dict['inputs'].unsqueeze(0).cuda()
with torch.no_grad():
    feat_maps = model.extract_feat(pre_inputs_tensor)
    outs = model.neck(feat_maps)
    head_out = model.bbox_head(outs)
    # head_out, _, _ = model.bbox_head.forward_single(outs)
    pred_heatmap = head_out[0][0].sigmoid()
print("预测Heatmap已生成。")

# --- 生成Ground Truth Heatmap ---
print("开始生成GT Heatmap...")
train_json_path = os.path.join(data_root, 'annotations', 'train.json')
val_json_path = os.path.join(data_root, 'annotations', 'val.json')
gt_bboxes_list_from_json = find_bboxes_for_image(image_name, [train_json_path, val_json_path])

if not gt_bboxes_list_from_json:
    print(f"无法为 {image_name} 生成GT Heatmap，因为没有找到任何标注。")
else:
    gt_bboxes_np = np.array(gt_bboxes_list_from_json, dtype=np.float32)
    gt_centroids_np = None
    precise_centroids_available = False  # <--- 引入标志位

    # ======================================================================
    #  核心改动: 尝试计算精确质心，如果失败则回退到几何中心
    # ======================================================================
    try:
        # --- 步骤 1: 尝试获取并使用 CalculateCentroids ---
        print("尝试通过注册表获取并使用 CalculateCentroids 来计算精确质心...")
        CentroidCalculatorClass = TRANSFORMS.get('CalculateCentroids')
        if CentroidCalculatorClass is None:
            raise RuntimeError(
                "无法从TRANSFORMS注册表中找到 'CalculateCentroids'。"
                "请确保配置文件中的 'custom_imports' 正确无误。")

        # --- 步骤 2: 实例化并准备输入 ---
        centroid_calculator = CentroidCalculatorClass()
        mock_bboxes = MockBoxes(torch.from_numpy(gt_bboxes_np))
        results_for_transform = {
            'img': img_rgb,
            'gt_bboxes': mock_bboxes,
            'img_path': img_path
        }

        # --- 步骤 3: 调用 transform 方法 ---
        transformed_results = centroid_calculator.transform(results_for_transform)

        # --- 步骤 4: 检查并提取质心 ---
        if 'gt_centroids' not in transformed_results or transformed_results['gt_centroids'] is None:
            raise ValueError("'gt_centroids' 键不存在或为空，无法获取精确质心。")

        gt_centroids_np = transformed_results['gt_centroids']
        precise_centroids_available = True  # <--- 成功！设置标志位
        print(f"精确质心计算完毕，共 {len(gt_centroids_np)} 个。将使用能够处理精确质心的get_targets版本。")

    except Exception as e:
        # --- 回退逻辑: 如果上述任何步骤失败 ---
        print(f"\n警告: 计算精确质心失败 (错误: {e})。")
        print(">>> 正在回退到使用标注框的几何中心作为高斯均值。 <<<\n")

        precise_centroids_available = False  # <--- 失败，确保标志位为False

        # gt_bboxes_np 的格式是 [x1, y1, x2, y2]
        x1 = gt_bboxes_np[:, 0]
        y1 = gt_bboxes_np[:, 1]
        x2 = gt_bboxes_np[:, 2]
        y2 = gt_bboxes_np[:, 3]

        # 计算几何中心 (centers_x, centers_y)
        centers_x = (x1 + x2) / 2.0
        centers_y = (y1 + y2) / 2.0

        # 将它们堆叠成一个 (N, 2) 的Numpy数组
        gt_centroids_np = np.stack((centers_x, centers_y), axis=-1)
        print(f"已计算 {len(gt_centroids_np)} 个几何中心。将使用标准版的get_targets。")
    # ======================================================================

    # --- 准备 DataSample ---
    gt_labels_np = np.zeros(len(gt_bboxes_np), dtype=np.int64)
    gt_instances = InstanceData()
    gt_instances.bboxes = torch.from_numpy(gt_bboxes_np).cuda()
    gt_instances.labels = torch.from_numpy(gt_labels_np).cuda()

    # 无论质心来源如何，都将其添加到 gt_instances 中。
    # 如果使用新版 get_targets，它会读取此值。
    # 如果使用旧版，此值会被忽略。
    gt_instances.gt_centroids = gt_centroids_np

    data_sample = DetDataSample()
    data_sample.gt_instances = gt_instances
    data_sample.img_shape = img_rgb.shape[:2]

    batch_data_samples = [data_sample]
    feat_shape = outs[0].shape

    # --- 根据标志位，动态调用不同版本的 get_targets ---
    with torch.no_grad():
        if precise_centroids_available:
            # 情况1: 精确质心可用，调用能够处理DataSample的新版get_targets
            # 签名: def get_targets(self, batch_data_samples, feat_shape)
            print("调用新版 get_targets(batch_data_samples, feat_shape)...")
            target_result, avg_factor = model.bbox_head.get_targets(
                batch_data_samples, feat_shape)
        else:
            # 情况2: 精确质心不可用，回退到标准get_targets
            # 签名: def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape)
            print("调用标准版 get_targets(gt_bboxes, gt_labels, feat_shape, img_shape)...")
            gt_bboxes_list = [ds.gt_instances.bboxes for ds in batch_data_samples]
            gt_labels_list = [ds.gt_instances.labels for ds in batch_data_samples]
            img_shape = data_sample.img_shape

            target_result, avg_factor = model.bbox_head.get_targets(
                gt_bboxes_list, gt_labels_list, feat_shape, img_shape)

    gt_heatmap_target = target_result['center_heatmap_target']
    print("GT Heatmap已生成。")

    # --- 可视化与保存 ---
    vis = Visualizer()
    vis_pred_heatmap = vis.draw_featmap(pred_heatmap.squeeze(0).cpu(), img.copy(), channel_reduction='select_max',
                                        resize_shape=(img.shape[0], img.shape[1]), alpha=0.4, topk=3)
    vis_gt_heatmap = vis.draw_featmap(gt_heatmap_target.squeeze(0).cpu(), img.copy(), channel_reduction='squeeze_mean',
                                      resize_shape=(img.shape[0], img.shape[1]), alpha=0.4, topk=3)

    comparison_img = np.hstack((vis_pred_heatmap, vis_gt_heatmap))

    # --- 添加文字标签 ---
    pil_img = Image.fromarray(cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman
        font = ImageFont.truetype(font_path, 15)
    except IOError:
        print(f"警告：无法在 {font_path} 找到字体文件，将使用默认字体。")
        font = ImageFont.load_default()

    draw.text((10, 10), "Pre-Heatmap", font=font, fill=(0, 255, 0))
    draw.text((vis_pred_heatmap.shape[1] + 10, 10), "GT-Heatmap", font=font, fill=(0, 0, 255))

    # comparison_img_final = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    comparison_img_final = np.array(pil_img)

    # --- 保存最终图像 ---
    save_dir = r'F:/ITD/vis_featuremap'
    os.makedirs(save_dir, exist_ok=True)
    formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_file = os.path.join(save_dir, f"{formatted_time}_{cfg_file_name}_comparison_{image_name}")

    cv2.imwrite(save_file, comparison_img_final)
    print(f"对比可视化图已保存到 {save_file}")
