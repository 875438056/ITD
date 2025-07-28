# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from YOLOv8Head
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         reduce_mean)
from mmdet.models.layers.bbox_nms import multiclass_nms

# 共享模块，建议放在 mmdet/models/layers/yolo_bricks.py
from ..backbones.yolo11_Backbone import Conv
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead


@MODELS.register_module()
class YOLOv11HeadModule(BaseModule):
    """YOLOv11HeadModule
    The Head part of the YOLOv11 model, which includes classification and regression branches.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, List],  # <--- 从配置中接收 [32, 64, 128, 256]
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Tuple[int] = (8, 16, 32),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(featmap_strides)
        self.num_base_priors = num_base_priors

        # In YOLOv11, the regression branch has 4 (for bbox) * 16 (for DFL) channels
        self.reg_max = 16  # DFL channels
        # 修复：proj_conv应该是1D卷积用于DFL的线性组合
        self.proj_conv = nn.Conv1d(self.reg_max, 1, 1, bias=False)

        # 初始化proj_conv的权重为线性递增序列
        nn.init.constant_(self.proj_conv.weight, 0.0)
        self.proj_conv.weight.data = torch.arange(self.reg_max, dtype=torch.float).view(1, self.reg_max, 1)

        # 修复：确保in_channels是列表
        if isinstance(in_channels, int):
            in_channels = [in_channels] * self.num_levels

        # w = lambda x: math.ceil(x * widen_factor / 8) * 8
        # self.in_channels = [w(c) for c in in_channels]

        self.in_channels = in_channels

        self._init_layers()

    def _init_layers(self):
        # 分别为 分类(cls) 和 回归(reg) 创建独立的卷积层列表
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # 分别为 分类(cls) 和 回归(reg) 创建独立的最终预测层列表
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for i in range(self.num_levels):
            # Stems
            self.cls_convs.append(
                self._build_stem(self.in_channels[i], self.in_channels[i]))
            self.reg_convs.append(
                self._build_stem(self.in_channels[i], self.in_channels[i]))

            # Prediction layers
            self.cls_preds.append(
                nn.Conv2d(self.in_channels[i], self.num_classes * self.num_base_priors, 1))
            self.reg_preds.append(
                nn.Conv2d(self.in_channels[i], 4 * self.reg_max, 1))

    def _build_stem(self, in_c, out_c):
        # A simple stem with two convs
        return nn.Sequential(
            Conv(in_c, out_c, 3),
            Conv(out_c, out_c, 3)
        )

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[List[Tensor], ...]:
        cls_scores = []
        bbox_preds = []
        for i, x in enumerate(feats):
            # Classification branch
            cls_feat = self.cls_convs[i](x)
            cls_score = self.cls_preds[i](cls_feat)
            cls_scores.append(cls_score)

            # Regression branch
            reg_feat = self.reg_convs[i](x)
            reg_pred = self.reg_preds[i](reg_feat)
            bbox_preds.append(reg_pred)

        return cls_scores, bbox_preds

    def integral(self, x: Tensor) -> Tensor:
        """DFL积分函数 - 将分布转换为精确坐标"""
        # 修复：处理不同的输入维度
        shape = x.shape
        if len(shape) == 3:  # [N, num_priors, 4*reg_max]
            x = x.reshape(shape[0], shape[1], 4, self.reg_max)
        elif len(shape) == 2:  # [N, 4*reg_max]
            x = x.reshape(shape[0], 4, self.reg_max)
        else:
            raise ValueError(f"Unsupported input shape: {shape}")

        # 在最后一个维度上应用softmax
        x = F.softmax(x, dim=-1)

        # 转换为conv1d期望的格式并应用积分
        # x shape: [batch_size, ..., 4, reg_max] -> [batch_size*...*4, reg_max, 1]
        original_shape = x.shape
        x_flat = x.reshape(-1, self.reg_max, 1)
        x_integrated = self.proj_conv(x_flat).squeeze(-1)  # [batch_size*...*4, 1] -> [batch_size*...*4]

        # 恢复原始形状
        result_shape = original_shape[:-1]  # 去掉最后的reg_max维度
        x_integrated = x_integrated.reshape(result_shape)

        return x_integrated


@MODELS.register_module()
class YOLOv11Head(BaseDenseHead):
    """YOLOv11Head
    The main head class that handles loss calculation and prediction.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType,
                 bbox_coder: ConfigType,
                 loss_cls: ConfigType,
                 loss_bbox: ConfigType,
                 loss_dfl: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)

        self.head_module = MODELS.build(head_module)
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.assigner = TASK_UTILS.build(train_cfg['assigner'])

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dfl = MODELS.build(loss_dfl)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.featmap_strides = self.prior_generator.strides
        self.num_classes = self.head_module.num_classes

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[List[Tensor], ...]:
        """Forward pass through the head module."""
        return self.head_module(x)


    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptConfigType = None) -> dict:

        # 1. Get priors and flatten predictions
        priors = self.prior_generator.grid_priors(
            [feat.shape[2:] for feat in cls_scores],
            device=cls_scores[0].device,
            with_stride=True)
        flatten_priors = torch.cat(priors)  # 把4个size的feature map展平成一维拼接到一起[128,64,32,16]**2=21760

        flatten_cls_preds = torch.cat([
            cls.permute(0, 2, 3, 1).reshape(len(batch_img_metas), -1, self.num_classes)
            for cls in cls_scores
        ], dim=1)

        # 预测四条边的距离（l, t, r, b），由于模型采用了一种更先进、更稳健的方法（DFL），
        # 它不直接预测一个精确的距离值，而是预测这个距离值的概率分布
        # 16: 这是你模型中的一个超参数，即 self.reg_max = 16。它代表概率分布的长度
        # 4: 代表边界框的4个变量，也就是我们熟悉的 l, t, r, b（到左、上、右、下四条边的距离）
        # 所以最后预测的是16*4=64

        flatten_bbox_preds = torch.cat([
            bbox.permute(0, 2, 3, 1).reshape(len(batch_img_metas), -1, 4 * self.head_module.reg_max)
            for bbox in bbox_preds
        ], dim=1)

        # 2. Decode bbox predictions (batched)
        batch_size = flatten_cls_preds.size(0)
        expanded_priors = flatten_priors.unsqueeze(0).expand(batch_size, -1, -1)
        # integral计算16个位置与其对应概率相乘得到一个浮点预测坐标，运算四次，解码得到边界框坐标信息LRTB(64-->4)
        pred_dist = self.head_module.integral(flatten_bbox_preds)
        # expanded_prior[..., :2]是先验的中心坐标，pred_dist是LRTB，两者解码得到box坐标（xyxy）
        decoded_bboxes = self.bbox_coder.decode(expanded_priors[..., :2], pred_dist)

        # 3. Label Assignment (Per-image in a loop)
        assign_results = []
        for i in range(batch_size):
            # ------------------- 核心修复开始 -------------------
            # 为每张图片创建一个干净的、不含批次维度的 pred_instances
            # InstanceData
            pred_instances_i = InstanceData()
            # 直接从批处理张量中切片出第 i 张图片的数据
            pred_instances_i.bboxes = decoded_bboxes[i].detach()
            pred_instances_i.scores = flatten_cls_preds[i].detach().sigmoid()
            # priors 是共享的，不含批次维度
            pred_instances_i.priors = flatten_priors

            gt_instances_i = batch_gt_instances[i]
            gt_instances_ignore_i = batch_gt_instances_ignore[i]

            assign_result_i = self.assigner.assign(
                pred_instances=pred_instances_i,
                gt_instances=gt_instances_i,
                gt_instances_ignore=gt_instances_ignore_i)
            # ------------------- 核心修复结束 -------------------
            assign_results.append(assign_result_i) # gt_inds:它告诉你每一个预测框（总共21760个）被分配给了哪个真实目标框（Ground Truth, GT）

        # 4. Concatenate results and calculate losses (此部分与上一版相同)
        # 将一个batch中的所有图片都拼在一起21760*24=522240
        gt_inds = torch.cat([res.gt_inds for res in assign_results])
        gt_labels = torch.cat([res.labels for res in assign_results])
        gt_scores = torch.cat([res.assign_metrics for res in assign_results])

        flatten_cls_preds = flatten_cls_preds.reshape(-1, self.num_classes).squeeze(-1) # (24,21760,1) --> (522240,1)
        flatten_bbox_preds = flatten_bbox_preds.reshape(-1, 4 * self.head_module.reg_max)  # (24,21760,64) --> (522240,64)
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)  # 4(xyxy)
        flatten_priors = expanded_priors.reshape(-1, 4) # 4(cx,cy,stride,stride)

        bbox_targets = torch.zeros_like(decoded_bboxes)
        bbox_weights = torch.zeros_like(decoded_bboxes)
        fg_mask = gt_inds > 0

        if fg_mask.any():
            all_gt_bboxes = torch.cat([gt.bboxes for gt in batch_gt_instances], dim=0)
            pos_assigned_gt_inds = gt_inds[fg_mask] - 1
            num_gts_per_img = [len(gt.bboxes) for gt in batch_gt_instances]
            img_idx = torch.cat([torch.full_like(res.gt_inds, i) for i, res in enumerate(assign_results)])
            pos_img_idx = img_idx[fg_mask]
            base_ind = torch.cumsum(torch.tensor([0] + num_gts_per_img[:-1]), dim=0).to(gt_inds.device)
            pos_assigned_gt_inds += base_ind[pos_img_idx]
            fg_bbox_targets = all_gt_bboxes[pos_assigned_gt_inds]
            bbox_targets[fg_mask, :] = fg_bbox_targets
            bbox_weights[fg_mask, :] = 1.0

        # loss_cls = self.loss_cls(flatten_cls_preds, gt_labels, gt_scores).sum()
        loss_cls = self.loss_cls(flatten_cls_preds, gt_scores).sum()
        # 初始化回归损失（loss_bbox）和DFL损失（loss_dfl）的值为0，并确保这个值为0的张量位于正确的计算设备上（CPU或GPU）
        loss_bbox = torch.tensor(0.0, device=cls_scores[0].device)
        loss_dfl = torch.tensor(0.0, device=cls_scores[0].device)

        if fg_mask.any():
            loss_bbox, loss_dfl = self._get_loss_bbox_dfl(
                flatten_priors[fg_mask],
                flatten_bbox_preds[fg_mask],
                bbox_targets[fg_mask],
                bbox_weights[fg_mask])

        num_pos = max(fg_mask.sum().float(), 1.0)
        loss_cls /= num_pos
        # loss_bbox /= num_pos
        # loss_dfl /= num_pos

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

    def _get_loss_bbox_dfl(self,
                           priors,  # [num_pos, 4]
                           bbox_preds,  # [num_pos, 64]
                           bbox_targets,  # [num_pos, 4]
                           bbox_weights):  # [num_pos, 4]
        """计算bbox和DFL损失"""
        # priors 和 assigned_priors 在这里是同一个东西，我们简化一下
        if priors.shape[0] == 0:
            return torch.zeros(1, device=priors.device), torch.zeros(1, device=priors.device)

        # 解码预测的bbox
        pred_bboxes = self.bbox_coder.decode(priors[..., :2],
                                             self.head_module.integral(bbox_preds))

        # 计算 avg_factor
        avg_factor = bbox_weights.sum()

        # --- 核心修复：直接传递 bbox_weights ---
        # 计算bbox损失。bbox_weights 的形状 [num_pos, 4] 与 pred_bboxes 一致，直接传入即可
        loss_bbox = self.loss_bbox(
            pred_bboxes,
            bbox_targets,
            weight=bbox_weights,  # 直接使用，不再 .sum()
            avg_factor=avg_factor)
        # --- 修复结束 ---

        # 计算DFL损失
        target_ltrb = self.bbox_coder.encode(priors[..., :2], bbox_targets, self.head_module.reg_max - 1)

        # 重新整形为DFL期望的格式
        pred_dist = bbox_preds.reshape(-1, 4, self.head_module.reg_max)

        loss_dfl = self.loss_dfl(
            pred_dist.reshape(-1, self.head_module.reg_max),
            target_ltrb.reshape(-1),
            weight=bbox_weights.reshape(-1),  # DFL损失的权重需要展平，这里是正确的
            avg_factor=avg_factor * 4)

        return loss_bbox, loss_dfl

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:

        cfg = self.test_cfg if cfg is None else cfg

        results_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_list = [score[img_id] for score in cls_scores]
            bbox_pred_list = [pred[img_id] for pred in bbox_preds]

            priors = self.prior_generator.grid_priors(
                [feat.shape[1:] for feat in cls_score_list],
                device=cls_scores[0].device)

            # Flatten and decode
            flatten_cls_preds = torch.cat(
                [cls.permute(1, 2, 0).reshape(-1, self.num_classes) for cls in cls_score_list])
            flatten_bbox_preds = torch.cat(
                [bbox.permute(1, 2, 0).reshape(-1, 4 * self.head_module.reg_max) for bbox in bbox_pred_list])
            flatten_priors = torch.cat(priors)

            decoded_bboxes = self.bbox_coder.decode(flatten_priors[..., :2],
                                                    self.head_module.integral(flatten_bbox_preds))

            # 修复：处理缩放
            if rescale:
                img_shape = batch_img_metas[img_id]['img_shape']
                ori_shape = batch_img_metas[img_id]['ori_shape']
                scale_factor = [ori_shape[1] / img_shape[1], ori_shape[0] / img_shape[0]] * 2
                decoded_bboxes = decoded_bboxes * decoded_bboxes.new_tensor(scale_factor)

            # NMS
            if with_nms:
                results = multiclass_nms(
                    decoded_bboxes,
                    flatten_cls_preds.sigmoid(),
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img)
            else:
                # 不使用NMS时返回原始结果
                results = InstanceData()
                results.bboxes = decoded_bboxes
                results.scores = flatten_cls_preds.sigmoid().max(dim=1)[0]
                results.labels = flatten_cls_preds.sigmoid().argmax(dim=1)

            results_list.append(results)

        return results_list