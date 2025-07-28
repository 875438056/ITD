# F:/ITD/models/transforms/centroid_transforms.py

import cv2
import numpy as np
from mmdet.registry import TRANSFORMS
from skimage.filters import threshold_otsu
from mmcv.transforms.base import BaseTransform


# F:/ITD/models/transforms/centroid_transforms.py
# 注册到数据处理组件 TRANSFORMS 是一个 Registry 对象，它管理了所有数据增强流程中可能用到的类或函数。
@TRANSFORMS.register_module()
class CalculateCentroids(BaseTransform):
    """一个自定义的数据处理模块，用于计算每个边界框内的质心。"""

    def __init__(self,
                 area_threshold: int = 50,
                 expansion_pixels: int = 2,
                 otsu_ratio: float = 0.6,
                 min_obj_size: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.area_threshold = area_threshold
        self.expansion_pixels = expansion_pixels
        self.otsu_ratio = otsu_ratio
        self.min_obj_size = min_obj_size

    def transform(self, results: dict) -> dict:
        """核心变换逻辑（装甲升级版）"""
        gt_centroids = []
        img = results['img']
        h, w = img.shape[:2]

        if 'gt_bboxes' not in results or results['gt_bboxes'].tensor.numel() == 0:
            results['gt_centroids'] = np.empty((0, 2), dtype=np.float32)
            return results

        # 【【【核心修正1：添加万能的try-except】】】
        for i, gt_bbox in enumerate(results['gt_bboxes'].tensor.numpy()):
            try:
                # 你的所有原始逻辑都放在try块内部
                x1, y1, x2, y2 = gt_bbox
                width = max(x1 - x2, x2 - x1)
                height = max(y1 - y2, y2 - y1)
                gt_bbox_area = width * height

                # 增加对NaN和inf的检查
                if np.any(np.isnan(gt_bbox)) or np.any(np.isinf(gt_bbox)):
                    # 如果坐标有问题，直接使用后备方案
                    gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])
                    continue

                if x1 >= x2 or y1 >= y2:
                    gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])
                    continue

                if gt_bbox_area <= self.area_threshold:
                    gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])
                else:
                    E = self.expansion_pixels
                    otsu_threshold_ratio = self.otsu_ratio
                    min_object_size = self.min_obj_size

                    # s1：获取扩展区域
                    x1_exp, y1_exp = int(max(0, x1 - E)), int(max(0, y1 - E))
                    x2_exp, y2_exp = int(min(w, x2 + E)), int(min(h, y2 + E))

                    patch = img[y1_exp:y2_exp, x1_exp:x2_exp]

                    if patch.size == 0:
                        gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])
                        continue

                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

                    min_val, max_val = np.min(patch_gray), np.max(patch_gray)
                    if max_val == min_val:
                        gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])
                        continue

                    # s2：归一化扩展区域
                    normalized_roi = (patch_gray - min_val) / (max_val - min_val)

                    # s3：计算目标和背景阈值，初步区分前背景
                    # Otsu的try-except保持
                    try:
                        t1 = threshold_otsu(normalized_roi) * otsu_threshold_ratio
                    except ValueError:
                        t1 = np.mean(normalized_roi)
                    t2 = np.mean(normalized_roi) + np.var(normalized_roi)
                    t = max(t1, t2)

                    binary_img = (normalized_roi > t).astype(np.uint8)

                    # s4：上下左右4连通性计算，找到均值最大，面积最大的连通区域
                    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=4)

                    # s5：计算目标mask
                    mask = np.zeros_like(patch_gray, dtype=np.uint8)
                    if num_labels > 1:
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        if areas.size > 0:
                            max_label_idx = np.argmax(areas) + 1
                            if stats[max_label_idx, cv2.CC_STAT_AREA] >= min_object_size:
                                mask = (labels_im == max_label_idx).astype(np.uint8)

                    if np.sum(mask) == 0:
                        gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])
                        continue

                    # s6：计算目标掩码区域的总能量强度
                    rows, cols = np.where(mask > 0)
                    intensities = patch_gray[rows, cols].astype(np.float64) + 1e-7
                    total_intensity = np.sum(intensities)

                    if total_intensity == 0:
                        gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])
                        continue

                    # s7：计算根据每个像素能量加权的质心坐标
                    mu_x_local = np.sum(cols * intensities) / total_intensity
                    mu_y_local = np.sum(rows * intensities) / total_intensity

                    # 原图尺度上的质心坐标
                    gt_centroids.append([x1_exp + mu_x_local, y1_exp + mu_y_local])

            except Exception as e:
                # 如果发生任何未预料的错误，打印信息并使用后备方案
                print(f"\n---!!! 在 CalculateCentroids 中捕获到未知异常 !!!---")
                print(f"图片路径: {results.get('img_path', 'N/A')}")
                print(f"出错的 BBox 索引: {i}, BBox 内容: {gt_bbox}")
                print(f"异常信息: {e}")
                print("--- 使用几何中心作为后备方案 ---")

                # 确保即使出错，也添加一个质心，保证数量匹配
                x1, y1, x2, y2 = gt_bbox
                gt_centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])

        # 【【【核心修正2：确保返回的空数组形状正确】】】
        if not gt_centroids:
            # 如果列表为空，返回一个 shape=(0, 2) 的数组
            results['gt_centroids'] = np.empty((0, 2), dtype=np.float32)
        else:
            results['gt_centroids'] = np.array(gt_centroids, dtype=np.float32)

        # 【【【最终保险：检查长度是否匹配】】】
        assert len(results['gt_centroids']) == len(results['gt_bboxes']), \
            f"质心数量 ({len(results['gt_centroids'])}) 与 BBox 数量 ({len(results['gt_bboxes'])}) 不匹配！"

        return results