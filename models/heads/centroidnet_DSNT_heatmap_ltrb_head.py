# F:/ITD/models/heads/centernet_heatmap_ltrb_offset_head.py

# ... (import anweisungen) ...
# 导入 kornia 的 DSNT 函数
from kornia.geometry.subpix import spatial_soft_argmax2d

import torch
from torch import Tensor
import numpy as np
from typing import List, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from mmdet.models.utils import (gaussian_radius, gen_gaussian_target, get_local_maximum,
                     get_topk_from_heatmap, multi_apply,
                     transpose_and_gather_feat)
from mmdet.registry import MODELS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead


# 辅助函数：从整张热力图中为每个点提取局部patch
def extract_local_patches(heatmap: Tensor, coords: Tensor, patch_size: int = 3) -> Tensor:
    """
    一个更通用的、完全向量化的patch提取函数。

    Args:
        heatmap (Tensor): 形状为 (N, C, H, W) 的热图，N是patch的数量。
        coords (Tensor): 形状为 (N, 2) 的坐标，包含 (y, x)。
        patch_size (int): patch的边长。

    Returns:
        Tensor: (N, C, patch_size, patch_size)
    """
    N, C, H, W = heatmap.shape
    half_size = patch_size // 2

    # 1. 生成一个基础的采样网格
    # torch.meshgrid现在需要indexing='ij'来保持旧的行为
    patch_grid_y, patch_grid_x = torch.meshgrid(
        torch.linspace(-half_size, half_size - 1, steps=patch_size, device=heatmap.device),
        torch.linspace(-half_size, half_size - 1, steps=patch_size, device=heatmap.device),
        indexing='ij'
    )
    # (patch_size, patch_size, 2) -> (1, patch_size, patch_size, 2)
    base_grid = torch.stack([patch_grid_x, patch_grid_y], dim=-1).unsqueeze(0)

    # 2. 将基础网格移动到每个坐标中心
    # coords (N, 2) -> (N, 1, 1, 2)
    # sampling_grid (N, patch_size, patch_size, 2)
    sampling_grid = coords.view(N, 1, 1, 2) + base_grid

    # 3. 将采样网格归一化到 [-1, 1] 以供 grid_sample 使用
    sampling_grid[..., 0] /= (W - 1) / 2
    sampling_grid[..., 1] /= (H - 1) / 2
    sampling_grid -= 1

    # 4. 【【【核心】】】直接进行grid_sample，无需循环
    # F.grid_sample 会将 sampling_grid 中的第i个网格应用在 heatmap 的第i个元素上
    patches = F.grid_sample(
        heatmap,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return patches


@MODELS.register_module()
class LTRBCentroidNetDSNTHead(BaseDenseHead):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channels (int): Number of channel in the input feature map.
        feat_channels (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (:obj:`ConfigDict` or dict): Config of center
            heatmap loss. Defaults to
            dict(type='GaussianFocalLoss', loss_weight=1.0)
        loss_ltrb (:obj:`ConfigDict` or dict): Config of wh loss. Defaults to
             dict(type='L1Loss', loss_weight=0.1).
        loss_centroid (:obj:`ConfigDict` or dict): Config of center loss. Defaults to
            dict(type='L1Loss', loss_weight=0.1).
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Useless in CenterNet, but we keep this variable for
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization
            config dict.
    """

    # 1. 修改 __init__ 方法
    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 num_classes: int,
                 radius_min: float,
                 patch_size: int,
                 loss_center_heatmap: ConfigType = dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_ltrb: ConfigType = dict(type='GIoULoss', loss_weight=1.0),
                 # 【【【新】】】为DSNT的坐标损失添加配置
                 loss_centroid: ConfigType = dict(type='L1Loss', loss_weight=0.1),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channels, feat_channels, num_classes)
        self.ltrb_head = self._build_head(in_channels, feat_channels, 4)
        # 【【【移除】】】 offset_head 不再需要
        # self.offset_head = self._build_head(in_channels, feat_channels, 2)

        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_ltrb = MODELS.build(loss_ltrb)
        # 【【【移除】】】 loss_offset 不再需要
        # self.loss_offset = MODELS.build(loss_offset)
        # 【【【新】】】构建新的坐标损失
        self.loss_centroid = MODELS.build(loss_centroid)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.radius_min = radius_min

    def _build_head(self, in_channels: int, feat_channels: int,
                    out_channels: int) -> nn.Sequential:
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.Mish(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1))
        return layer

    # 2. 修改 init_weights 方法
    def init_weights(self) -> None:
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        # 【【【移除】】】 从列表中移除 offset_head
        for head in [self.ltrb_head]:  # 只初始化 ltrb_head
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    # 3. 修改 forward 和 forward_single 方法
    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[List[Tensor]]:
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            ltrb_preds (list[Tensor]): ltrb predicts for all levels, the channels
                number is 4.
            centroid_preds (list[Tensor]): offset predicts for all levels, the
               channels number is N(num_target).
        """
        # 【【【移除】】】不再返回 offset_preds
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        ltrb_pred = self.ltrb_head(x)
        # 【【【移除】】】不再预测 offset_pred
        # offset_pred = self.offset_head(x)
        return center_heatmap_pred, ltrb_pred  # 只返回两个预测

    def get_targets(self,
                    batch_gt_instances: List['InstanceData'],
                    batch_img_metas: List[dict],
                    feat_shape: Tuple[int, ...]) -> Tuple[dict, float]:
        bs, _, feat_h, feat_w = feat_shape
        device = batch_gt_instances[0].bboxes.device

        center_heatmap_target = torch.zeros([bs, self.num_classes, feat_h, feat_w], device=device)
        ltrb_target = torch.zeros([bs, 4, feat_h, feat_w], device=device)

        gt_scaled_coords, gt_indices, gt_ltrb_for_loss = [], [], []
        valid_gt_mask = []  # 【新】用于标记有效的GT

        for batch_id in range(bs):
            gt_instances = batch_gt_instances[batch_id]
            img_meta = batch_img_metas[batch_id]
            img_h, img_w = img_meta['img_shape']
            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels

            if gt_bboxes.size(0) == 0:
                continue

            gt_centroids_numpy = gt_instances.gt_centroids
            gt_centroids_orig = torch.from_numpy(gt_centroids_numpy).to(device)

            width_ratio = float(feat_w / img_w)
            height_ratio = float(feat_h / img_h)
            scale_tensor = torch.tensor([width_ratio, height_ratio], device=device)
            scaled_gt_centroids = gt_centroids_orig * scale_tensor

            for j, scaled_ct in enumerate(scaled_gt_centroids):
                ctx_int, cty_int = scaled_ct.int()

                # --- 1. 生成所有GT的信息 ---
                # ... (这部分逻辑和原来完全一样) ...
                x1, y1, x2, y2 = gt_bboxes[j]
                scale_box_h = (y2 - y1) * height_ratio
                scale_box_w = (x2 - x1) * width_ratio
                radius = gaussian_radius((scale_box_h, scale_box_w), min_overlap=0.7)
                radius = max(self.radius_min, int(radius))
                gt_label = gt_labels[j]
                # 【修改】我们只对有效的点生成热图
                # gen_gaussian_target(center_heatmap_target[batch_id, gt_label], [ctx_int, cty_int], radius)
                centroid_x_orig, centroid_y_orig = gt_centroids_orig[j]
                l = (centroid_x_orig - x1) * width_ratio
                t = (centroid_y_orig - y1) * height_ratio
                r = (x2 - centroid_x_orig) * width_ratio
                b = (y2 - centroid_y_orig) * height_ratio

                # --- 2. 【新】检查GT是否有效 ---
                is_valid = (0 <= ctx_int < feat_w and 0 <= cty_int < feat_h)
                valid_gt_mask.append(is_valid)

                if is_valid:
                    # 只为有效的GT生成target
                    gen_gaussian_target(center_heatmap_target[batch_id, gt_label], [ctx_int, cty_int], radius)
                    ltrb_target[batch_id, :, cty_int, ctx_int] = torch.tensor([l, t, r, b], device=device)

                # --- 3. 【修改】无论是否有效，都先收集，后面再过滤 ---
                gt_scaled_coords.append(scaled_ct)
                gt_indices.append(torch.tensor([batch_id, gt_label, cty_int, ctx_int], device=device))
                gt_ltrb_for_loss.append(torch.tensor([l, t, r, b], device=device))

        avg_factor = max(1, center_heatmap_target.eq(1).sum())

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            gt_scaled_coords=torch.stack(gt_scaled_coords) if gt_scaled_coords else torch.empty(0, 2, device=device),
            gt_indices=torch.stack(gt_indices).long() if gt_indices else torch.empty(0, 4, device=device),
            gt_ltrb_for_loss=torch.stack(gt_ltrb_for_loss) if gt_ltrb_for_loss else torch.empty(0, 4, device=device),
            valid_gt_mask=torch.tensor(valid_gt_mask, dtype=torch.bool,
                                       device=device) if valid_gt_mask else torch.empty(0, dtype=torch.bool,
                                                                                        device=device)
        )
        return target_result, avg_factor

    # 2. 重写 loss 方法，这是核心
    # 在 LTRBCentroidNetDSNTHead 类中

    # 【【【第1步：添加一个标准的 loss 入口方法】】】
    def loss(self, x: Tuple[Tensor], batch_data_samples: List['DetDataSample']) -> dict:
        """
        这个方法是框架直接调用的入口。
        它的职责是调用 forward 得到预测，然后将预测和标签一起传给 loss_by_feat。
        """
        outs = self(x)  # 调用 forward()，得到 (center_heatmap_preds, ltrb_preds)

        # 从 data_samples 中解包出 Gt annd ImgMeta 信息
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_gt_instances.append(data_sample.gt_instances)
            batch_img_metas.append(data_sample.metainfo)

        # 准备 loss_by_feat 所需的全部参数
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)

        # 调用 loss_by_feat
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    # 【【【第2步：修改 loss_by_feat 的函数签名】】】
    # 在 LTRBCentroidNetDSNTHead 类中
    def loss_by_feat(self,
                     center_heatmap_preds: List[Tensor],
                     ltrb_preds: List[Tensor],
                     batch_gt_instances: List['InstanceData'],
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:

        center_heatmap_pred = center_heatmap_preds[0]
        ltrb_pred = ltrb_preds[0]
        feat_shape = center_heatmap_pred.shape
        device = center_heatmap_pred.device

        target_result, avg_factor = self.get_targets(
            batch_gt_instances, batch_img_metas, feat_shape)

        center_heatmap_target = target_result['center_heatmap_target']
        gt_scaled_coords = target_result['gt_scaled_coords'].to(device)
        gt_indices = target_result['gt_indices'].to(device)  # [batch_id, gt_label, cty_int, ctx_int]
        gt_ltrb = target_result['gt_ltrb_for_loss'].to(device)
        valid_gt_mask = target_result['valid_gt_mask'].to(device)  # 获取有效掩码

        num_total_targets = gt_scaled_coords.shape[0]

        # 【修改】avg_factor现在基于有效目标的数量
        num_valid_targets = valid_gt_mask.sum().item()
        avg_factor_ltrb_centroid = max(1, num_valid_targets)

        # 1. 热图辅助损失 (不变，它基于渲染的高斯，已经是正确的)
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)

        # 如果没有任何有效目标，直接返回
        if num_valid_targets == 0:
            return dict(
                loss_center_heatmap=loss_center_heatmap,
                loss_ltrb=torch.tensor(0., device=device),
                loss_centroid=torch.tensor(0., device=device))

        # 【【【核心：使用掩码过滤所有参与损失计算的数据】】】
        valid_indices = gt_indices[valid_gt_mask]
        valid_gt_ltrb = gt_ltrb[valid_gt_mask]
        valid_gt_scaled_coords = gt_scaled_coords[valid_gt_mask]

        # 2. LTRB 损失
        pred_ltrb = ltrb_pred[valid_indices[:, 0], :, valid_indices[:, 2], valid_indices[:, 3]]
        loss_ltrb = self.loss_ltrb(pred_ltrb, valid_gt_ltrb, avg_factor=avg_factor_ltrb_centroid)

        # 3. DSNT 质心坐标损失
        gt_batch_ids = valid_indices[:, 0]
        gt_class_ids = valid_indices[:, 1]
        # coords现在只需要(y,x)
        # patch_center_yx_coords = valid_indices[:, 2:4]  # (N, 2)
        patch_center_xy_coords = valid_indices[:, [3, 2]].float()  # (N, 2), 现在是(x, y)顺序

        heatmap_for_each_gt = center_heatmap_pred[gt_batch_ids, gt_class_ids]

        local_patches = extract_local_patches(
            heatmap_for_each_gt.unsqueeze(1), # (N, 1, H, W)
            patch_center_xy_coords.float(), # 传入浮点型的(y,x)坐标
            patch_size=self.patch_size)# 返回 (N, 1, 16, 16)

        pred_coords_norm = spatial_soft_argmax2d(
            local_patches, normalized_coordinates=True).squeeze(1)

        true_offset_pixels = valid_gt_scaled_coords - valid_indices[:, [3, 2]].float()
        true_coords_norm = true_offset_pixels / (self.patch_size / 2.)

        # # ==================== DEBUGGING BLOCK START ====================
        # print("-" * 50)
        # print(f"DEBUG INFO: Batch with {num_valid_targets} valid targets.")
        # print(f"Shape of pred_coords_norm (Prediction): {pred_coords_norm.shape}")
        # print(f"Shape of true_coords_norm (Target):   {true_coords_norm.shape}")
        #
        # # 进一步检查，如果形状的数字一样，但维度不同
        # if pred_coords_norm.dim() != true_coords_norm.dim():
        #     print(
        #         f"!!! CRITICAL: Dimension mismatch! Pred_dim={pred_coords_norm.dim()}, True_dim={true_coords_norm.dim()}")
        #
        # # 如果形状不匹配，我们看看它们的来源
        # if pred_coords_norm.shape != true_coords_norm.shape:
        #     print("!!! SHAPES MISMATCH! Inspecting sources...")
        #     print(f"Source shape for prediction (local_patches): {local_patches.shape}")
        #     print(f"Source shape for target (valid_gt_scaled_coords): {valid_gt_scaled_coords.shape}")
        # print("-" * 50)
        # # ===================== DEBUGGING BLOCK END =====================

        loss_centroid = self.loss_centroid(
            pred_coords_norm, true_coords_norm, avg_factor=avg_factor_ltrb_centroid)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_ltrb=loss_ltrb,
            loss_centroid=loss_centroid)

    def predict_by_feat(self,
                        center_heatmap_preds: List[Tensor],
                        ltrb_preds: List[Tensor],
                        # 【移除】 offset_preds 不再是输入参数
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = True,
                        with_nms: bool = False) -> InstanceList:
        """
        Transform network output for a batch into bbox predictions.
        (将一个批次的网络输出转换为边界框预测)
        """
        # 断言输入数量匹配
        assert len(center_heatmap_preds) == len(ltrb_preds) == 1

        result_list = []
        for img_id in range(len(batch_img_metas)):
            # 【修改】调用 _predict_by_feat_single 时，不再传入 offset_pred
            result_list.append(
                self._predict_by_feat_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    ltrb_preds[0][img_id:img_id + 1, ...],
                    batch_img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _predict_by_feat_single(self,
                                center_heatmap_pred: Tensor,
                                ltrb_pred: Tensor,
                                # 【移除】 offset_pred 不再是输入参数
                                img_meta: dict,
                                rescale: bool = True,
                                with_nms: bool = False) -> InstanceData:
        """
        Transform outputs of a single image into bbox results.
        (将单张图片的输出转换为边界框结果)
        """
        # 【修改】调用 _decode_heatmap 时，不再传入 offset_pred
        batch_det_bboxes, batch_labels = self._decode_heatmap(
            center_heatmap_pred,
            ltrb_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        # --- 后续的NMS和rescale逻辑保持不变 ---
        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        if 'border' in img_meta:  # MMDetection 3.x compatibility
            batch_border = det_bboxes.new_tensor(img_meta['border'])[..., [2, 0, 2, 0]]
            det_bboxes[..., :4] -= batch_border

        if rescale and 'scale_factor' in img_meta:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if with_nms:
            # Batched NMS in MMDetection requires a dummy class_id when using class-agnostic NMS
            # For simplicity, we assume class-agnostic NMS if test_cfg.nms is a dict without 'multi_class'
            # This part can be complex, make sure it matches your MMDetection version
            keep = batched_nms(det_bboxes[:, :4], det_bboxes[:, 4], det_labels, self.test_cfg.nms)[1]
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]
            if 'max_per_img' in self.test_cfg:
                det_bboxes = det_bboxes[:self.test_cfg.max_per_img]
                det_labels = det_labels[:self.test_cfg.max_per_img]

        results = InstanceData()
        results.bboxes = det_bboxes[..., :4]
        results.scores = det_bboxes[..., 4]
        results.labels = det_labels
        return results


    # 在 LTRBCentroidNetDSNTHead 类中
    def _decode_heatmap(self,
                        center_heatmap_pred: Tensor,
                        ltrb_pred: Tensor,
                        img_shape: tuple,
                        k: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor]:

        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape
        b = center_heatmap_pred.shape[0]

        # 1. 找到所有峰值点 (不变)
        center_heatmap_pred_nms = get_local_maximum(center_heatmap_pred, kernel=kernel)
        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred_nms, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        # 2. 【【【修正后的DSNT精炼坐标】】】
        # a. 准备提取patch所需的数据
        batch_ids = torch.arange(b, device=topk_ys.device).unsqueeze(1).repeat(1, k)
        class_ids = batch_topk_labels.view(-1)
        peak_xy_coords = torch.stack([topk_xs.view(-1), topk_ys.view(-1)], dim=1).float() # (B*k, 2), (x,y)顺序

        # peak_coords: (B*k, 3) with (batch_id, y, x)
        # peak_yx_coords = torch.stack([topk_ys.view(-1), topk_xs.view(-1)], dim=1)
        # class_ids: (B*k,)


        # b. 提取每个峰值对应类别的单通道热力图
        #    这里我们创建一个索引，从 (B, C, H, W) 中选出 (B*k, H, W) 的热图
        # heatmap_for_peaks = center_heatmap_pred[peak_coords[:, 0], class_ids]

        heatmap_for_peaks = center_heatmap_pred[batch_ids.view(-1), class_ids]

        # c. 提取patch

        local_patches = extract_local_patches(
            heatmap_for_peaks.unsqueeze(1),  # (B*k, 1, H, W)
            peak_xy_coords.float(), # 传入浮点型的(y,x)坐标
            patch_size=self.patch_size
        ) # (B*k, patch_size, patch_size)

        # d. 计算亚像素偏移
        offsets_norm = spatial_soft_argmax2d(local_patches, normalized_coordinates=True)
        offsets_pixels = offsets_norm * (self.patch_size / 2.)
        squeeze_offsets_pixels = offsets_pixels.squeeze(1) # 形状从 (B*k, 1, 2) 变为 (B*k, 2)

        # e. 将偏移加回到原始整数坐标上
        refined_xs = topk_xs.view(-1) + squeeze_offsets_pixels[:, 0]
        refined_ys = topk_ys.view(-1) + squeeze_offsets_pixels[:, 1]

        # 3. 提取ltrb预测 (不变)
        ltrb = transpose_and_gather_feat(ltrb_pred, batch_index)

        # 4. 解码成bbox (使用精炼后的坐标)
        # ... (这部分逻辑和原来一样) ...
        radio_w = float(inp_w / width)
        radio_h = float(inp_h / height)

        tl_x = (refined_xs - ltrb.view(-1, 4)[:, 0]) * radio_w
        tl_y = (refined_ys - ltrb.view(-1, 4)[:, 1]) * radio_h
        br_x = (refined_xs + ltrb.view(-1, 4)[:, 2]) * radio_w
        br_y = (refined_ys + ltrb.view(-1, 4)[:, 3]) * radio_h

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=1).view(b, k, 4)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores.unsqueeze(-1)), dim=-1)

        return batch_bboxes, batch_topk_labels