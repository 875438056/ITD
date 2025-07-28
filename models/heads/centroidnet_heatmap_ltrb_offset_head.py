# 文件路径: F:/ITD/models/heads/custom_centernet_head.py
import torch
from typing import List, Tuple
from torch import Tensor
import numpy as np

from mmdet.registry import MODELS
# 假设 LTRBCenterNetHead 已经存在于下述路径
from models.heads.centernet_heatmap_ltrb_offset_head import LTRBCenterNetHead
from mmdet.models.utils import (gaussian_radius, gen_gaussian_target, get_local_maximum,
                                get_topk_from_heatmap, multi_apply,
                                transpose_and_gather_feat)


@MODELS.register_module()
class LTRBCentroidNetHead(LTRBCenterNetHead):
    """
    一个终极版的自定义Head，它实现了：
    1. 以自定义的“质心”（Centroid）作为检测中心。
    2. 以 l,t,r,b（中心到四边距离）的方式回归边界框。
    """

    def get_targets(self,
                    batch_data_samples: List['DetDataSample'],
                    feat_shape: Tuple[int, ...]) -> Tuple[dict, float]:
        """
        重写 get_targets 方法。
        核心改动：使用传入的 gt_centroids 代替 bbox 的几何中心。
        """
        bs, _, feat_h, feat_w = feat_shape
        device = batch_data_samples[0].gt_instances.bboxes.device

        # 准备好需要填充的目标张量
        center_heatmap_target = torch.zeros(
            [bs, self.num_classes, feat_h, feat_w], device=device)
        ltrb_target = torch.zeros([bs, 4, feat_h, feat_w], device=device)
        offset_target = torch.zeros([bs, 2, feat_h, feat_w], device=device)
        # 权重张量可以被 ltrb 和 offset 损失共用
        target_weight = torch.zeros([bs, 1, feat_h, feat_w], device=device)

        for batch_id in range(bs):
            data_sample = batch_data_samples[batch_id]
            gt_instances = data_sample.gt_instances
            img_h, img_w = data_sample.img_shape

            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels

            # 如果没有真实物体，直接跳到下一张图
            if gt_bboxes.size(0) == 0:
                continue

            # 【【【核心逻辑：使用自定义质心】】】
            # 1. 从 data_sample 中获取自定义质心
            # gt_centroids_numpy = gt_instances.gt_centroids
            # gt_centroids_orig = torch.from_numpy(gt_centroids_numpy).to(device)
            gt_centroids_orig = torch.from_numpy(gt_instances.gt_centroids).to(device)

            # 2. 将原图尺度的质心，缩放到特征图尺度
            width_ratio = float(feat_w / img_w)
            height_ratio = float(feat_h / img_h)
            scale_tensor = torch.tensor([width_ratio, height_ratio], device=device)
            scaled_gt_centroids = gt_centroids_orig * scale_tensor

            # 遍历每一个物体
            for j, scaled_ct in enumerate(scaled_gt_centroids):
                # 获取整数化的质心坐标，用于定位
                ctx_int, cty_int = scaled_ct.int() # 截断向零取整

                # 安全边界检查，防止中心点在数据增强后超出特征图范围
                if not (0 <= ctx_int < feat_w and 0 <= cty_int < feat_h):
                    continue

                # --- 准备用于生成target的各种数据 ---
                gt_bbox = gt_bboxes[j]
                x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
                # 这是原图尺度下的质心
                centroid_x_orig, centroid_y_orig = gt_centroids_orig[j]

                # --- 1. 生成高斯热图目标 (Heatmap Target) ---
                # 使用bbox的宽高来计算高斯半径，这更稳定
                scale_box_h = (y2_gt - y1_gt) * height_ratio
                scale_box_w = (x2_gt - x1_gt) * width_ratio
                radius = gaussian_radius((scale_box_h, scale_box_w), min_overlap=0.7)
                radius = max(self.radius_min, int(radius))
                gt_label = gt_labels[j]

                # return: out_heatmap
                gen_gaussian_target(center_heatmap_target[batch_id, gt_label],
                                    [ctx_int, cty_int], radius)

                # --- 2. 生成 l,t,r,b 目标 ---
                # 计算方法：ltrb = 质心到四边的距离，特征图尺度下的ltrb
                l = (centroid_x_orig - x1_gt) * width_ratio
                t = (centroid_y_orig - y1_gt) * height_ratio
                r = (x2_gt - centroid_x_orig) * width_ratio
                b = (y2_gt - centroid_y_orig) * height_ratio
                #
                # # 注意：这里我们直接回归距离值，FCOS等方法有时会除以一个步长(stride)，这里简化处理
                ltrb_target[batch_id, 0, cty_int, ctx_int] = l
                ltrb_target[batch_id, 1, cty_int, ctx_int] = t
                ltrb_target[batch_id, 2, cty_int, ctx_int] = r
                ltrb_target[batch_id, 3, cty_int, ctx_int] = b

                # # 对数编码
                # epsilon = 1e-6  # 防止log(0)
                # # 将 l,t,r,b 四个0维张量堆叠成一个 1x4 的张量
                # ltrb = torch.stack([l, t, r, b])
                # # 使用 clamp 和 torch.log 进行向量化操作
                # ltrb_log = torch.log(ltrb.clamp(min=epsilon))

                # 将计算结果分别填入 target
                # ltrb_target[batch_id, 0, cty_int, ctx_int] = ltrb_log[0]  # l
                # ltrb_target[batch_id, 1, cty_int, ctx_int] = ltrb_log[1]  # t
                # ltrb_target[batch_id, 2, cty_int, ctx_int] = ltrb_log[2]  # r
                # ltrb_target[batch_id, 3, cty_int, ctx_int] = ltrb_log[3]  # b


                # --- 3. 生成中心点偏移目标 (Offset Target) ---
                # 目标是质心浮点坐标相对于整数坐标的偏移
                ctx_float, cty_float = scaled_ct
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx_float - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty_float - cty_int

                # --- 4. 生成权重 (Weight) ---
                # 在该中心点位置，ltrb和offset的损失都需要计算
                target_weight[batch_id, 0, cty_int, ctx_int] = 1

        # 计算平均因子，用于平衡loss
        avg_factor = max(1, center_heatmap_target.eq(1).sum())

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            ltrb_target=ltrb_target,
            offset_target=offset_target,
            target_weight=target_weight)  # 返回统一的权重

        return target_result, avg_factor

    # loss 方法需要重写，以适配新的target_result字典和权重用法
    def loss(self, x: Tuple[Tensor],
             batch_data_samples: List['DetDataSample']) -> dict:

        outs = self(x)
        # forward返回的是一个元组的列表，每个元素对应一个特征层
        # 对于无FPN的CenterNet，我们只取第一个（也是唯一一个）
        center_heatmap_preds, ltrb_preds, offset_preds = outs

        center_heatmap_pred = center_heatmap_preds[0]
        ltrb_pred = ltrb_preds[0]
        offset_pred = offset_preds[0]

        feat_shape = center_heatmap_pred.shape

        target_result, avg_factor = self.get_targets(batch_data_samples, feat_shape)

        center_heatmap_target = target_result['center_heatmap_target']
        ltrb_target = target_result['ltrb_target']
        offset_target = target_result['offset_target']
        target_weight = target_result['target_weight']  # 使用新的统一权重

        # 计算 heatmap loss
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)

        # 计算 ltrb loss
        # 我们只计算有目标的位置的loss (target_weight > 0)
        # avg_factor_ltrb = target_weight.sum()
        loss_ltrb = self.loss_ltrb(
            ltrb_pred,
            ltrb_target,
            weight=target_weight.expand(-1, 4, -1, -1),  # 将权重从1通道扩展到4通道
            avg_factor=avg_factor * 4)  # 沿用之前*2的约定

        # 计算 offset loss
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            weight=target_weight.expand(-1, 2, -1, -1),  # 将权重从1通道扩展到2通道
            avg_factor=avg_factor * 2)  # offset loss通常不乘系数

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_ltrb=loss_ltrb,
            loss_offset=loss_offset)

    def _decode_heatmap(self,
                        center_heatmap_pred: Tensor,
                        # 【修改15】接收 ltrb_pred
                        ltrb_pred: Tensor,
                        offset_pred: Tensor,
                        img_shape: tuple,
                        k: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor]:
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (tuple): image shape in hw format.
            k (int): Get top k center keypoints from heatmap. Defaults to 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Defaults to 3.

        Returns:
            tuple[Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        # ... get_topk_from_heatmap 的逻辑不变 ...
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        # 【修改16】从 ltrb_pred 中提取特征
        ltrb = transpose_and_gather_feat(ltrb_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        # 计算feature map的stride，用于还原bbox坐标到原图尺度
        radio_w = float(inp_w / width)
        radio_h = float(inp_h / height)

        # 【修改17】使用 l,t,r,b 解码 bbox
        # ltrb 的4个通道分别对应 l,t,r,b 预测
        # 注意：ltrb是featuremap尺度的预测值，不是直接的距离

        # l_decoded = torch.exp(ltrb[..., 0])
        # t_decoded = torch.exp(ltrb[..., 1])
        # r_decoded = torch.exp(ltrb[..., 2])
        # b_decoded = torch.exp(ltrb[..., 3])
        #
        # tl_x = (topk_xs - l_decoded) * radio_w
        # tl_y = (topk_ys - t_decoded) * radio_h
        # br_x = (topk_xs + r_decoded) * radio_w
        # br_y = (topk_ys + b_decoded) * radio_h

        tl_x = (topk_xs - ltrb[..., 0]) * radio_w
        tl_y = (topk_ys - ltrb[..., 1]) * radio_h
        br_x = (topk_xs + ltrb[..., 2]) * radio_w
        br_y = (topk_ys + ltrb[..., 3]) * radio_h


        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels