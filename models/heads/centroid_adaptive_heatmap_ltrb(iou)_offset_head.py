# 文件路径: F:/ITD/models/heads/custom_centernet_head.py
import torch
from typing import List, Tuple
from torch import Tensor
import numpy as np
from scipy.special import erfinv

from mmdet.registry import MODELS
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
# 假设 LTRBCenterNetHead 已经存在于下述路径
from models.heads.centernet_heatmap_ltrb_offset_head import LTRBCenterNetHead


@MODELS.register_module()
class AdaptiveLTRBIoUCentroidNetHead(LTRBCenterNetHead):
    """
    自定义Head，它实现了：
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

        # --- 为了高效计算，预先生成特征图坐标网格 ---
        # y_coords 和 x_coords 的 shape 均为 (feat_h, feat_w)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(feat_h, device=device),
            torch.arange(feat_w, device=device),
            indexing='ij')

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
            gt_centroids_numpy = gt_instances.gt_centroids
            gt_centroids_orig = torch.from_numpy(gt_centroids_numpy).to(device)

            # 2. 将原图尺度的质心，缩放到特征图尺度
            width_ratio = float(feat_w / img_w)
            height_ratio = float(feat_h / img_h)
            scale_tensor = torch.tensor([width_ratio, height_ratio], device=device)
            scaled_gt_centroids = gt_centroids_orig * scale_tensor

            # 遍历每一个物体
            for j, scaled_ct in enumerate(scaled_gt_centroids):
                # 获取整数化的质心坐标，用于定位
                ctx_int, cty_int = scaled_ct.int() # 截断向零取整
                ctx_float, cty_float = scaled_ct

                # 安全边界检查，防止中心点在数据增强后超出特征图范围
                if not (0 <= ctx_int < feat_w and 0 <= cty_int < feat_h):
                    continue

                # --- 准备用于生成target的各种数据 ---
                gt_bbox = gt_bboxes[j]
                gt_label = gt_labels[j]
                x1, y1, x2, y2 = gt_bbox
                # 这是原图尺度下的质心
                centroid_x_orig, centroid_y_orig = gt_centroids_orig[j]

                # --- 1. 生成高斯热图目标 (Heatmap Target) ---
                # 使用bbox的宽高来计算高斯半径，这更稳定
                # scale_box_h = (y2 - y1) * height_ratio
                # scale_box_w = (x2 - x1) * width_ratio
                # radius = gaussian_radius((scale_box_h, scale_box_w), min_overlap=0.7)
                # radius = max(0, int(radius))
                # gt_label = gt_labels[j]
                #
                # # return: out_heatmap
                # gen_gaussian_target(center_heatmap_target[batch_id, gt_label],
                #                     [ctx_int, cty_int], radius)

                # gt_box原图尺度下的(gt_box_w,gt_box_h) --> 缩放到特征图尺度(scale_box_w,scale_box_h)
                # heatmap = np.zeros((feat_h, feat_w), dtype=np.float32)
                scale_box_h = (y2 - y1) * height_ratio
                scale_box_w = (x2 - x1) * width_ratio

                # 如果边界框无效 (宽或高 <= 0)，则返回空的热力图
                if scale_box_w <= 0 or scale_box_h <= 0:
                    continue

                # --- 核心逻辑：根据论文的公式 ---
                # 1. 根据热力图上边界框的面积，确定参数 P 的值
                area_scaled_bbox = scale_box_w * scale_box_h
                if area_scaled_bbox <= 6:
                    P = 0.65
                elif area_scaled_bbox <= 12:
                    P = 0.65
                else:
                    P = 0.99
                # P = 0.99

                # 2. 使用公式 (9) 计算 k
                k = torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(torch.tensor(P, device=device))

                # 3. 使用公式 (10) 计算 sigma_x 和 sigma_y
                sigma_x = scale_box_w / (2 * k) + 1e-4
                sigma_y = scale_box_h / (2 * k) + 1e-4

                # sigma_x = max(1, sigma_x)
                # sigma_y = max(1, sigma_y)

                # =================================================================== #
                # 【【【核心改动：计算高斯边界并裁剪】】】

                # # d. 根据3σ原则确定高斯核的有效范围（边界框）
                # radius_x, radius_y = 3 * sigma_x, 3 * sigma_y
                # # 使用浮点中心确定边界，然后转换为整数
                # left = int(ctx_float - radius_x)
                # right = int(ctx_float + radius_x) + 1  # +1 是因为切片不包含右边界
                # top = int(cty_float - radius_y)
                # bottom = int(cty_float + radius_y) + 1
                #
                # # e. 将边界框裁剪到特征图范围内，确保不越界
                # clipped_left = max(0, left)
                # clipped_right = min(feat_w, right)
                # clipped_top = max(0, top)
                # clipped_bottom = min(feat_h, bottom)
                #
                # # 如果裁剪后的区域无效（例如，整个高斯核都在特征图外），则跳过
                # if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
                #     continue
                #
                # # f. 只在裁剪后的有效区域内生成高斯热图
                # # 从预先生成的网格中，切出对应的小块区域
                # patch_x_coords = x_coords[clipped_top:clipped_bottom, clipped_left:clipped_right]
                # patch_y_coords = y_coords[clipped_top:clipped_bottom, clipped_left:clipped_right]
                #
                # # 在这个小块上计算高斯值
                # exponent = (((patch_x_coords - ctx_float) ** 2) / (2 * sigma_x ** 2) +
                #             ((patch_y_coords - cty_float) ** 2) / (2 * sigma_y ** 2))
                # patch_heatmap = torch.exp(-exponent)
                #
                # # g. 将生成的小块热图绘制到最终目标张量的对应位置上
                # target_slice = center_heatmap_target[batch_id, gt_label, clipped_top:clipped_bottom,
                #                clipped_left:clipped_right]
                # center_heatmap_target[batch_id, gt_label, clipped_top:clipped_bottom,
                # clipped_left:clipped_right] = torch.maximum(
                #     target_slice, patch_heatmap)
                # # =================================================================== #

                # 4. 生成单个高斯热图
                # G(x, y) = exp(-[ ((x-μx)^2 / (2σx^2)) + ((y-μy)^2 / (2σy^2)) ])
                # μx, μy 是浮点坐标，以实现亚像素精度，备用ctx_int, cty_int
                # x_coords, y_coords 是预先生成的网格
                # exponent = (((x_coords - ctx_float)**2) / (2 * sigma_x**2) +
                #             ((y_coords - cty_float)**2) / (2 * sigma_y**2))
                exponent = (((x_coords - ctx_int) ** 2) / (2 * sigma_x ** 2) +
                            ((y_coords - cty_int) ** 2) / (2 * sigma_y ** 2))
                single_heatmap = torch.exp(-exponent)



                # 5. 将生成的热图绘制到最终目标张量上
                # 关键：使用 torch.maximum 来正确处理重叠区域，保留峰值
                # [batch_id, gt_label] 精准定位到需要绘制的通道
                # center_heatmap_target[bs, self.num_classes, feat_h, feat_w]
                target_slice = center_heatmap_target[batch_id, gt_label]
                center_heatmap_target[batch_id, gt_label] = torch.maximum(
                    target_slice, single_heatmap)


                # --- 2. 生成 l,t,r,b 目标 ---
                # 计算方法：ltrb = 质心到四边的距离
                l = (centroid_x_orig - x1) * width_ratio
                t = (centroid_y_orig - y1) * height_ratio
                r = (x2 - centroid_x_orig) * width_ratio
                b = (y2 - centroid_y_orig) * height_ratio

                # 注意：这里我们回归距离值时会除以一个步长(stride)
                ltrb_target[batch_id, 0, cty_int, ctx_int] = l
                ltrb_target[batch_id, 1, cty_int, ctx_int] = t
                ltrb_target[batch_id, 2, cty_int, ctx_int] = r
                ltrb_target[batch_id, 3, cty_int, ctx_int] = b

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

    # 文件路径: F:/ITD/models/heads/custom_centernet_head.py


    # 在 AdaptiveLTRBCentroidNetHead 类中
    def loss(self, x: Tuple[Tensor],
             batch_data_samples: List['DetDataSample']) -> dict:

        outs = self(x)
        center_heatmap_preds, ltrb_preds, offset_preds = outs

        # 通常只处理单尺度特征
        center_heatmap_pred = center_heatmap_preds[0]
        ltrb_pred = ltrb_preds[0]
        offset_pred = offset_preds[0]

        feat_shape = center_heatmap_pred.shape #(bs,1,feat_h,feat_w)
        org_img_shape = batch_data_samples[0].batch_input_shape # tuple (256,256)

        stride_h, stride_w = (org_img_shape[0] / feat_shape[2]), (org_img_shape[1] / feat_shape[3])

        if stride_w == stride_h:
            stride = stride_h
        else:print('stride != stride_h')

        device = center_heatmap_pred.device

        # 1. 获取真值 (Targets)
        target_result, avg_factor = self.get_targets(batch_data_samples, feat_shape)
        center_heatmap_target = target_result['center_heatmap_target']
        ltrb_target = target_result['ltrb_target']
        offset_target = target_result['offset_target']
        target_weight = target_result['target_weight']  # 这个权重标记了正样本的位置

        # 2. 计算热图损失 (Heatmap Loss)
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)

        # 3. 核心修改：以可微分方式计算IoU损失

        # a. 找到所有正样本（gt objects）在特征图上的索引
        # target_weight 的 shape 是 (bs, 1, H, W)，>0 的地方就是正样本
        pos_mask = target_weight.squeeze(1) > 0  # 得到一个布尔掩码 (bs, H, W)

        # 如果当前批次没有正样本，则损失为0
        if not pos_mask.any():
            loss_ltrb = ltrb_pred.sum() * 0
            loss_offset = offset_pred.sum() * 0
            return dict(
                loss_center_heatmap=loss_center_heatmap,
                loss_ltrb=loss_ltrb,
                loss_offset=loss_offset)

        # b. 准备用于重建边界框的坐标网格
        # y_coords, x_coords 的 shape 均为 (feat_H, feat_W)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(feat_shape[2], device=device),
            torch.arange(feat_shape[3], device=device),
            indexing='ij')
        # 将坐标网格扩展到与批次大小匹配 (bs, feat_H, feat_W)
        batch_y_coords = y_coords.expand(feat_shape[0], -1, -1)
        batch_x_coords = x_coords.expand(feat_shape[0], -1, -1)

        # c. 重建【预测】边界框 (Predicted BBoxes)
        # 只在正样本位置进行操作
        pos_ltrb_pred = ltrb_pred.permute(0, 2, 3, 1)[pos_mask]  # (num_pos, 4)
        pos_offset_pred = offset_pred.permute(0, 2, 3, 1)[pos_mask]  # (num_pos, 2)
        pos_x_coords = batch_x_coords[pos_mask]  # (num_pos,)
        pos_y_coords = batch_y_coords[pos_mask]  # (num_pos,)

        # 预测的中心点 = 整数坐标 + 预测的偏移量
        pred_center_x = pos_x_coords + pos_offset_pred[:, 0]
        pred_center_y = pos_y_coords + pos_offset_pred[:, 1]

        # 解码为 (x1, y1, x2, y2)
        # l, t, r, b 分别在 pos_ltrb_pred 的 0, 1, 2, 3 通道
        pred_x1 = pred_center_x - pos_ltrb_pred[:, 0]
        pred_y1 = pred_center_y - pos_ltrb_pred[:, 1]
        pred_x2 = pred_center_x + pos_ltrb_pred[:, 2]
        pred_y2 = pred_center_y + pos_ltrb_pred[:, 3]
        pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # (num_pos, 4)

        # d. 重建【目标】边界框 (Target BBoxes)
        pos_ltrb_target = ltrb_target.permute(0, 2, 3, 1)[pos_mask]  # (num_pos, 4)
        pos_offset_target = offset_target.permute(0, 2, 3, 1)[pos_mask]  # (num_pos, 2)

        # 目标的中心点 = 整数坐标 + 目标的偏移量
        target_center_x = pos_x_coords + pos_offset_target[:, 0]
        target_center_y = pos_y_coords + pos_offset_target[:, 1]

        # 解码为 (x1, y1, x2, y2)
        target_x1 = target_center_x - pos_ltrb_target[:, 0]
        target_y1 = target_center_y - pos_ltrb_target[:, 1]
        target_x2 = target_center_x + pos_ltrb_target[:, 2]
        target_y2 = target_center_y + pos_ltrb_target[:, 3]
        target_boxes = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=1)  # (num_pos, 4)

        # 新增核心步骤：将边界框还原到原图尺度
        # 假设步长对于x和y方向是相同的

        # 将特征图尺度的坐标乘以步长，还原到原图尺度
        pred_boxes_orig_scale = pred_boxes * stride
        target_boxes_orig_scale = target_boxes * stride


        # e. 计算 IoU 损失
        # avg_factor 通常是正样本的数量
        num_pos = pos_mask.sum()
        # loss_ltrb = self.loss_ltrb(pred_boxes, target_boxes)/num_pos
        loss_ltrb = self.loss_ltrb(pred_boxes_orig_scale, target_boxes_orig_scale) / num_pos

        # 4. 计算偏移损失 (Offset Loss)，这部分不变，但通常也只对正样本计算
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            weight=target_weight.expand(-1, 2, -1, -1),
            avg_factor=num_pos * 2)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_ltrb=loss_ltrb,
            loss_offset=loss_offset)