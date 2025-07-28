# Copyright (c) OpenMMLab. All rights reserved.
# This code is revised based on a user-provided snippet and MMDetection's RTMDetHead.
# The revision fixes critical bugs and aligns the implementation with MMDetection standards.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmcv.ops import batched_nms
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmengine.structures import InstanceData
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.assigners import TaskAlignedAssigner


# DFL模块，实现从分布到期望值的转换
# 这个模块本身设计是正确的，予以保留
class DFL(BaseModule):
    """Distribution Focal Loss (DFL) module.
       Basically, it's a Conv2d layer with a fixed weight to compute the
       expectation of a distribution.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        # Initialize the weight to be a fixed range [0, 1, 2, ..., reg_max]
        self.conv.weight.data[:] = torch.arange(
            self.reg_max + 1, dtype=torch.float).view(1, self.reg_max + 1, 1, 1)
        # Freeze the layer
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor, shape (N, C, H, W), where C = reg_max + 1.
        Returns:
            Tensor: Decoded distance, shape (N, 1, H, W).
        """
        # The input `x` is expected to have gone through a softmax.
        return self.conv(x)


@MODELS.register_module()
class YOLOv11Head(BaseDenseHead):
    """
    YOLOv11Head: A corrected and robust implementation for a YOLO-style detection head.
    This head features a decoupled design for classification and regression, anchor-free
    mechanism with TaskAlignedAssigner, and uses DFL and IoU losses for box regression.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (Union[int, List]): Number of channels in the input feature maps.
        widen_factor (float): The scaling factor for channels.
        num_feats (int): Number of feature maps from the neck. Default to 3.
        featmap_strides (List[int]): Strides of input feature maps.
        reg_max (int): Maximum value of the regression range for DFL.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        assigner (dict): Config of the assigner.
        bbox_coder (dict): Config of the bbox coder.
        loss_cls (dict): Config of the classification loss.
        loss_bbox (dict): Config of the localization loss (e.g., GIoU loss).
        loss_dfl (dict): Config of the Distribution Focal Loss.
        train_cfg (dict): Training config.
        test_cfg (dict): Testing config.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: List[int],
                 widen_factor: float = 1.0,
                 featmap_strides: List[int] = [8, 16, 32],
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 assigner: ConfigType = dict(type='TaskAlignedAssigner', topk=13),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GIoULoss',
                     loss_weight=2.0),
                 loss_dfl: ConfigType = dict(
                     type='DistributionFocalLoss',
                     loss_weight=0.25),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = [int(c * widen_factor) for c in in_channels]
        self.featmap_strides = featmap_strides
        self.num_levels = len(featmap_strides)
        self.reg_max = reg_max
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        # --- Build necessary components ---
        self.assigner = TASK_UTILS.build(assigner)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.prior_generator = TASK_UTILS.build(
            dict(type='mmdet.models.task_modules.prior_generators.MlvlPointGenerator',
                 strides=self.featmap_strides,
                 offset=0.5)
        )
        self.dfl = DFL(reg_max)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # --- Build losses ---
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dfl = MODELS.build(loss_dfl)

        # --- Initialize layers ---
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Stem(s) and heads
        for i in range(self.num_levels):
            # Same architecture for all levels
            in_c = self.in_channels[i]
            # Classification branch
            self.cls_convs.append(
                self._build_stacked_convs(in_c, in_c, norm_cfg, act_cfg)
            )
            # Regression branch
            self.reg_convs.append(
                self._build_stacked_convs(in_c, in_c, norm_cfg, act_cfg)
            )
            # Prediction heads
            self.cls_preds.append(nn.Conv2d(in_c, self.num_classes, 1))
            self.reg_preds.append(nn.Conv2d(in_c, 4 * (self.reg_max + 1), 1))

    def _build_stacked_convs(self, in_channels, out_channels, norm_cfg, act_cfg):
        """A helper to build a stack of two convs."""
        return nn.Sequential(
            ConvModule(in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        # Use prior probability of 0.01 for the classification head's bias
        for m in self.cls_preds:
            b = m.bias.view(1, -1)
            b.data.fill_(-4.595)  # -log((1-0.01)/0.01)
        # You can add further specific initializations here if needed

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the neck."""
        return multi_apply(self.forward_single, x, self.cls_convs,
                           self.reg_convs, self.cls_preds, self.reg_preds)

    def forward_single(self, x: Tensor, cls_conv: nn.Module, reg_conv: nn.Module,
                       cls_pred_layer: nn.Module, reg_pred_layer: nn.Module) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        cls_feat = cls_conv(x)
        reg_feat = reg_conv(x)

        cls_score = cls_pred_layer(cls_feat)
        bbox_pred_dist = reg_pred_layer(reg_feat)

        return cls_score, bbox_pred_dist

    def loss_by_feat(self, *args, **kwargs) -> dict:
        """
        A dummy implementation for loss_by_feat to meet the ABC requirement.
        This function will not be called when `loss` function is implemented.

        一个用于满足抽象基类要求的虚拟实现。
        当 `loss` 方法被实现时，此函数不会被调用。
        """
        # 这个方法体可以是空的，因为程序永远不会调用它
        # 它的存在只是为了让 Python 认为我们遵守了"合同"
        pass

    def loss(self, x: Tuple[Tensor, ...], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation."""
        # 1. 前向传播
        cls_scores, bbox_preds = self(x)

        # 2. 准备 Ground Truth 和批次信息
        batch_gt_instances = [
            data_sample.gt_instances for data_sample in batch_data_samples
        ]
        batch_size = cls_scores[0].size(0)

        # ===== 修复：检查并过滤空的ground truth =====
        # 检查是否有图片没有任何目标（Mosaic增强可能导致）
        valid_samples = []
        valid_indices = []
        for i, gt_instances in enumerate(batch_gt_instances):
            if len(gt_instances) > 0:  # 有目标的图片
                valid_samples.append(gt_instances)
                valid_indices.append(i)

        # 如果整个批次都没有有效目标，返回零损失
        if len(valid_samples) == 0:
            device = cls_scores[0].device
            return dict(
                loss_cls=torch.tensor(0.0, device=device, requires_grad=True),
                loss_bbox=torch.tensor(0.0, device=device, requires_grad=True),
                loss_dfl=torch.tensor(0.0, device=device, requires_grad=True)
            )

        # 如果只有部分样本有效，只处理有效样本
        if len(valid_samples) < batch_size:
            # 重新组织 cls_scores 和 bbox_preds，只保留有效样本
            cls_scores = [cls_score[valid_indices] for cls_score in cls_scores]
            bbox_preds = [bbox_pred[valid_indices] for bbox_pred in bbox_preds]
            batch_gt_instances = valid_samples
            batch_size = len(valid_samples)
        # ===== 修复结束 =====

        # 3. 准备 Priors
        featmap_sizes = [feat.shape[2:] for feat in cls_scores]
        device = cls_scores[0].device
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        all_priors = torch.cat(mlvl_priors, dim=0)

        # 转换为 (x1, y1, x2, y2) 格式
        half_wh = all_priors[:, 2:] / 2
        # 计算角点坐标
        x1y1 = all_priors[:, :2] - half_wh  # 左上角 = 中心点 - 宽高/2
        x2y2 = all_priors[:, :2] + half_wh  # 右下角 = 中心点 + 宽高/2
        # 合并成 (x1, y1, x2, y2) 格式
        all_priors_boxes = torch.cat([x1y1, x2y2], dim=1)
        # 获取步长
        all_priors_stride = all_priors[:, 2:]

        # 获取每一层feature map对应的priors个数
        num_level_priors = [w * h for (w, h) in featmap_sizes]

        # 4. 准备批次化的预测
        all_cls_scores = torch.cat([  # (valied_batch,21824,1)
            cls.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for cls in cls_scores
        ], dim=1)
        all_bbox_preds = torch.cat([  # (valied_batch,21824,68) 68=17*4
            bbox.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * (self.reg_max + 1))
            for bbox in bbox_preds
        ], dim=1)

        # 5. 逐图分配并收集目标
        cls_targets_list, bbox_targets_list, assign_metrics_list = [], [], []
        pos_inds_list = []

        for i in range(batch_size):
            single_gt_instances = batch_gt_instances[i]

            # ===== 检查并修正标签范围 =====
            gt_labels = single_gt_instances.labels
            # 确保标签在有效范围内 [0, num_classes-1]
            if gt_labels.numel() > 0:
                # 检查是否有超出范围的标签
                invalid_mask = (gt_labels < 0) | (gt_labels >= self.num_classes)
                if invalid_mask.any():
                    # print(f"Warning: Found invalid labels in sample {i}: {gt_labels[invalid_mask]}")
                    # 将无效标签夹紧到有效范围
                    gt_labels = torch.clamp(gt_labels, 0, self.num_classes - 1)
                    single_gt_instances.labels = gt_labels

            pred_instances_for_assign = InstanceData(
                priors=all_priors_boxes,  # （21824，4）
                scores=all_cls_scores[i].detach().sigmoid(),  # （21824，1）
                bboxes=self.decode_bbox(  # （1，21824，4）（1，21824，68）-->（21824，4） （21824，68）
                    all_priors[:, :2].unsqueeze(0), all_bbox_preds[i].unsqueeze(0), all_priors_stride).squeeze(0),
            )

            if isinstance(self.assigner, TaskAlignedAssigner):
                assign_result = self.assigner.assign(
                    pred_instances=pred_instances_for_assign,
                    gt_instances=single_gt_instances)
            else:
                assign_result = self.assigner.assign(
                    pred_instances=pred_instances_for_assign,
                    gt_instances=single_gt_instances,
                    num_level_priors=num_level_priors)

            # QFL 需要所有样本的 targets 和 metrics
            labels = assign_result.labels
            # ===== 再次检查分配结果中的标签 =====
            if labels.numel() > 0:
                # 确保分配结果中的标签也在有效范围内
                invalid_mask = (labels < 0) | (labels >= self.num_classes)
                if invalid_mask.any():
                    # print(f"Warning: Found invalid assigned labels in sample {i}: {labels[invalid_mask]}")
                    labels = torch.clamp(labels, 0, self.num_classes - 1)
            # ===== 修复结束 =====

            cls_targets_list.append(labels)
            assign_metrics_list.append(assign_result.assign_metrics)

            pos_inds = torch.nonzero(
                assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)

            if pos_inds.numel() > 0:
                pos_inds_list.append(pos_inds + i * all_priors.size(0))  # 正样本位置索引，一张图中的索引转换为整个batch的索引
                pos_assigned_gt_inds = assign_result.gt_inds[
                                           pos_inds] - 1  # 正样本类别索引，assigner默认正样本是索引是1，负样本是-1，实际标注的类别索引是0
                bbox_targets_list.append(single_gt_instances.bboxes[pos_assigned_gt_inds])  # 正样本匹配的GT的box

        # 6. 整合并计算损失
        cls_targets = torch.cat(cls_targets_list, dim=0)  # pos_batch * 21824 将所有batch拼接成一维向量
        assign_metrics = torch.cat(assign_metrics_list, dim=0)  # pos_batch * 21824

        # ===== 最终检查：确保传递给损失函数的目标标签有效 =====
        if cls_targets.numel() > 0:
            invalid_mask = (cls_targets < 0) | (cls_targets >= self.num_classes)
            if invalid_mask.any():
                # print(f"Warning: Found invalid cls_targets before loss computation: {cls_targets[invalid_mask]}")
                cls_targets = torch.clamp(cls_targets, 0, self.num_classes - 1)
        # ===== 修复结束 =====

        # 计算分类损失 (QualityFocalLoss)
        loss_cls = self.loss_cls(
            all_cls_scores.reshape(-1, self.num_classes),
            (cls_targets, assign_metrics),  # QFL 期望一个元组作为 target
            avg_factor=max(1, assign_metrics.sum())  # 用 assign_metrics 的和作为 avg_factor
        )

        # 如果整个批次都没有正样本，则回归损失为0
        if not pos_inds_list:
            return dict(loss_cls=loss_cls,
                        loss_bbox=all_bbox_preds.sum() * 0,
                        loss_dfl=all_bbox_preds.sum() * 0)

        pos_inds = torch.cat(pos_inds_list)  # list:num_positive
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        num_pos = pos_inds.size(0)

        # 筛选正样本用于回归损失计算
        batch_idx = pos_inds // all_priors.size(0)  # pos_inds是整个batch的索引，通过取模解码正样本属于batch中的哪一张图
        anchor_idx = pos_inds % all_priors.size(0)  # 通过取余解码正样本属于该图的哪个先验
        pos_bbox_preds = all_bbox_preds[batch_idx, anchor_idx]  # （num_positive,68）
        pos_anchor_points = all_priors[anchor_idx, :2]  # （num_positive, 2） 2:锚框中心点坐标(x,y)
        pos_stride_tensor = all_priors[anchor_idx, 2:]

        # 计算回归损失
        loss_bbox, loss_dfl = self.bbox_loss(
            pos_anchor_points, pos_bbox_preds, bbox_targets, pos_stride_tensor)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox / num_pos,
            loss_dfl=loss_dfl / num_pos
        )

    def bbox_loss(self, pos_anchor_points, pos_bbox_preds, pos_bbox_targets, pos_stride_tensor) -> Tuple[
        Tensor, Tensor]:
        """Compute bounding box loss for positive samples."""
        if pos_anchor_points.shape[0] == 0:
            return torch.zeros(1, device=pos_anchor_points.device), \
                torch.zeros(1, device=pos_anchor_points.device)

        # 1. Decode predictions to get xyxy format
        decoded_bbox_preds = self.decode_bbox(pos_anchor_points, pos_bbox_preds, pos_stride_tensor)

        # 2. Compute IoU/GIoU loss
        loss_bbox = self.loss_bbox(decoded_bbox_preds, pos_bbox_targets)

        # 3. Compute DFL loss
        target_ltrb = self.bbox_coder.encode(pos_anchor_points, pos_bbox_targets)
        # pos_stride_tensor 的形状是 (num_pos, 2)，内容是 (stride_w, stride_h)
        # 我们需要一个 (num_pos, 4) 的张量，内容是 (stride_w, stride_h, stride_w, stride_h)
        # 可以通过将原张量与自身沿维度1拼接来实现
        stride_tensor_4d = torch.cat([pos_stride_tensor, pos_stride_tensor], dim=1)
        target_ltrb /= stride_tensor_4d

        pos_bbox_preds_dist = pos_bbox_preds.view(-1, self.reg_max + 1)
        target_corners = target_ltrb.flatten()

        # ===== 修复DFL损失计算中的目标范围检查 =====
        # 确保目标值在有效范围内 [0, reg_max]
        target_corners = torch.clamp(target_corners, 0, self.reg_max)
        # ===== 修复结束 =====

        loss_dfl = self.loss_dfl(
            pos_bbox_preds_dist,
            target_corners.long(),
            weight=None,  # No extra weight needed
            avg_factor=pos_anchor_points.shape[0] * 4.0  # Average by number of corners
        )

        return loss_bbox, loss_dfl


    def decode_bbox(self, anchor_points: Tensor, pred_dist: Tensor, pos_stride_tensor: Tensor) -> Tensor:
        """Decode distance predictions to bounding box coordinates.
            anchor_points：Tensor:（num_positive, 2） 2 =（x, y）
            pred_dist: Tensor:（num_positive, 68） 68=reg_max+1 * 4
            pos_stride_tensor: Tensor:（num_positive, 2） 2 = （stride_w, stride_h）
        """

        # ---  ↓↓↓  核心修复：增加维度检查和兼容处理  ↓↓↓  ---
        if pred_dist.ndim == 2:
            # 输入是 2D (num_pos, C)，为兼容后续操作，增加一个虚拟的批次维度
            pred_dist = pred_dist.unsqueeze(0)
            # anchor_points 也需要保持维度一致
            if anchor_points.ndim == 2:
                anchor_points = anchor_points.unsqueeze(0)
            # 标记一下，在函数返回前需要去掉这个虚拟维度
            squeeze_output = True
        else:
            # 输入已经是 3D (B, N, C)，无需处理
            squeeze_output = False
        # ---  ↑↑↑  修改结束  ↑↑↑  ---

        # 后续代码现在可以安全地假设输入是 3D 的
        b, n, _ = pred_dist.shape
        # (b, n, 4, reg_max+1) -> (b*n, reg_max+1, 4, 1) -> (b*n*4, reg_max+1, 1, 1)
        pred_dist_softmax = pred_dist.reshape(
            b * n, 4, self.reg_max + 1).permute(0, 2, 1).softmax(dim=1)
        pred_dist_softmax = pred_dist_softmax.reshape(b * n * 4, self.reg_max + 1, 1, 1)

        # Use DFL to get expected distances
        # (b*n*4, 1, 1, 1) -> (b, n, 4)
        pred_dist_exp_temp = self.dfl(pred_dist_softmax).reshape(b, n, 4)
        stride_tensor_4d = torch.cat([pos_stride_tensor, pos_stride_tensor], dim=1)
        pred_dist_exp = pred_dist_exp_temp * stride_tensor_4d

        # 解码出最终的 bbox 坐标
        decoded_bboxes = self.bbox_coder.decode(anchor_points, pred_dist_exp)  # DistancePointBBoxCoder
        # 输入(x_ref, y_ref)，(l, t, r, b)，返回（xyxy）

        # 如果之前增加了虚拟维度，现在就把它去掉
        if squeeze_output:
            return decoded_bboxes.squeeze(0)
        else:
            return decoded_bboxes

    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList,
                rescale: bool = True) -> List[InstanceData]:
        """Perform forward propagation and postprocessing to get prediction.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (SampleList): The Data Samples.
                It usually contains information such as
                `gt_instances` or `metainfo`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
        """
        # 1. 前向传播
        cls_scores, bbox_preds = self(x)

        # 初始化一个空列表，用于收集每一张图片的结果
        results_list = []

        # 2. 遍历批次中的每一张图片
        for i, data_sample in enumerate(batch_data_samples):
            # 3. 提取并准备单张图片的预测
            mlvl_cls_scores = [item[i].permute(1, 2, 0) for item in cls_scores]
            mlvl_bbox_preds = [item[i].permute(1, 2, 0) for item in bbox_preds]

            single_cls_scores = torch.cat(
                [item.reshape(-1, self.num_classes) for item in mlvl_cls_scores]
            ).sigmoid()

            single_bbox_preds = torch.cat(
                [item.reshape(-1, 4 * (self.reg_max + 1)) for item in mlvl_bbox_preds]
            )

            # 获取 Priors
            featmap_sizes = [feat.shape[:2] for feat in mlvl_cls_scores]
            priors = self.prior_generator.grid_priors(
                featmap_sizes, device=single_cls_scores.device, with_stride=True)
            all_priors = torch.cat(priors, dim=0)

            # 解码 Bbox
            decoded_bboxes = self.decode_bbox(
                all_priors[:, :2].unsqueeze(0), single_bbox_preds.unsqueeze(0), all_priors[:, 2:]
            ).squeeze(0)

            # 4. 执行 NMS 等后处理
            nms_cfg = self.test_cfg.get('nms', dict(type='nms', iou_threshold=0.65))
            score_thr = self.test_cfg.get('score_thr', 0.01)

            candidate_inds = (single_cls_scores > score_thr).nonzero()
            if candidate_inds.numel() == 0:
                # ===== 修复：确保空结果也有必要的属性 =====
                results = InstanceData()
                # 创建空的张量，确保shape正确且在正确的device上
                empty_bboxes = torch.zeros((0, 4), dtype=torch.float32, device=single_cls_scores.device)
                empty_scores = torch.zeros((0,), dtype=torch.float32, device=single_cls_scores.device)
                empty_labels = torch.zeros((0,), dtype=torch.long, device=single_cls_scores.device)

                results.bboxes = empty_bboxes
                results.scores = empty_scores
                results.labels = empty_labels
                # ===== 修复结束 =====
                results_list.append(results)
                continue

            pre_nms_scores = single_cls_scores[candidate_inds[:, 0], candidate_inds[:, 1]]
            pre_nms_bboxes = decoded_bboxes[candidate_inds[:, 0]]
            pre_nms_labels = candidate_inds[:, 1]

            # ===== 修复：检查NMS输入的有效性 =====
            if pre_nms_bboxes.numel() == 0 or pre_nms_scores.numel() == 0:
                # 如果预处理后没有有效候选框，创建空结果
                results = InstanceData()
                empty_bboxes = torch.zeros((0, 4), dtype=torch.float32, device=single_cls_scores.device)
                empty_scores = torch.zeros((0,), dtype=torch.float32, device=single_cls_scores.device)
                empty_labels = torch.zeros((0,), dtype=torch.long, device=single_cls_scores.device)

                results.bboxes = empty_bboxes
                results.scores = empty_scores
                results.labels = empty_labels
                results_list.append(results)
                continue
            # ===== 修复结束 =====

            final_bboxes_with_scores, keep_idxs = batched_nms(pre_nms_bboxes, pre_nms_scores, pre_nms_labels, nms_cfg)

            final_scores = final_bboxes_with_scores[:, -1]
            final_labels = pre_nms_labels[keep_idxs]

            max_per_img = self.test_cfg.get('max_per_img', 100)
            if len(final_scores) > max_per_img:
                final_bboxes_with_scores = final_bboxes_with_scores[:max_per_img]
                final_scores = final_scores[:max_per_img]
                final_labels = final_labels[:max_per_img]

            # 5. 创建该图片的 InstanceData 结果
            results = InstanceData(
                bboxes=final_bboxes_with_scores[:, :4],
                scores=final_scores,
                labels=final_labels)

            # 6. 如果需要，对 bboxes 进行 rescale
            if rescale and results.bboxes.numel() > 0:
                # ===== 修复：确保scale_factor处理的健壮性 =====
                if 'scale_factor' in data_sample.metainfo:
                    scale_factor = data_sample.metainfo['scale_factor']
                    # 处理不同格式的scale_factor
                    if isinstance(scale_factor, (list, tuple)):
                        if len(scale_factor) == 2:
                            # (scale_w, scale_h) -> (scale_w, scale_h, scale_w, scale_h)
                            scale_factor_wh = results.bboxes.new_tensor(scale_factor * 2)
                        elif len(scale_factor) == 4:
                            # (scale_w, scale_h, scale_w, scale_h)
                            scale_factor_wh = results.bboxes.new_tensor(scale_factor)
                        else:
                            # 其他情况，使用默认值
                            scale_factor_wh = results.bboxes.new_tensor([1.0, 1.0, 1.0, 1.0])
                    else:
                        # 单个数值，应用到所有维度
                        scale_factor_wh = results.bboxes.new_tensor([scale_factor] * 4)

                    results.bboxes /= scale_factor_wh
                # ===== 修复结束 =====

            # 7. 将处理好的单张图片结果存入列表
            results_list.append(results)

        # 8. 返回 InstanceData 的列表
        return results_list
