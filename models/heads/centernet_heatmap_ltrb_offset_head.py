# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
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
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
# from mmdet.models.dense_heads import centernet_head


@MODELS.register_module()
class LTRBCenterNetHead(BaseDenseHead):
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
        loss_wh (:obj:`ConfigDict` or dict): Config of wh loss. Defaults to
             dict(type='L1Loss', loss_weight=0.1).
        loss_offset (:obj:`ConfigDict` or dict): Config of offset loss.
            Defaults to dict(type='L1Loss', loss_weight=1.0).
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Useless in CenterNet, but we keep this variable for
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization
            config dict.
    """

    # 在 CenterNetHead 的 __init__ 方法中

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 num_classes: int,
                 radius_min: int,
                 loss_center_heatmap: ConfigType = dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 # 【修改1】将 loss_wh 重命名为 loss_ltrb，并可以选用更适合的损失函数，如 GIoULoss
                 loss_ltrb: ConfigType = dict(type='GIoULoss', loss_weight=1.0),
                 loss_offset: ConfigType = dict(
                     type='L1Loss', loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channels, feat_channels,
                                             num_classes)
        # 【修改2】将 wh_head 改为 ltrb_head，输出通道从 2 变为 4
        self.ltrb_head = self._build_head(in_channels, feat_channels, 4)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)

        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)
        # 【修改3】构建新的 loss_ltrb
        self.loss_ltrb = MODELS.build(loss_ltrb)
        self.loss_offset = MODELS.build(loss_offset)

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

        # 修改 CenterNetHead 的 init_weights 方法中 wh_head
    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        # 【修改4】将 self.wh_head 改为 self.ltrb_head
        for head in [self.ltrb_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

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
            offset_preds (list[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            ltrb_preds (Tensor): ltrb predicts, the channels number is 4.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        # 【修改5】将 wh_pred 改为 ltrb_pred
        ltrb_pred = self.ltrb_head(x)
        offset_pred = self.offset_head(x)
        return center_heatmap_pred, ltrb_pred, offset_pred


    def loss_by_feat(
            self,
            center_heatmap_preds: List[Tensor],
            # 【修改10】接收 ltrb_preds
            ltrb_preds: List[Tensor],
            offset_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:

        # 【修改11】断言和变量解包也要对应修改
        assert len(center_heatmap_preds) == len(ltrb_preds) == len(
            offset_preds) == 1

        center_heatmap_pred = center_heatmap_preds[0]
        ltrb_pred = ltrb_preds[0]
        offset_pred = offset_preds[0]

        # ... get_targets 的调用逻辑不变 ...
        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']
        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     center_heatmap_pred.shape,
                                                     img_shape)

        # 【修改12】从 target_result 中获取新的目标
        center_heatmap_target = target_result['center_heatmap_target']
        ltrb_target = target_result['ltrb_target']
        offset_target = target_result['offset_target']
        ltrb_offset_target_weight = target_result['ltrb_offset_target_weight']

        # heatmap loss 计算不变
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)

        # 【修改13】计算 ltrb 损失
        # 提取出有目标的点的 prediction 和 target
        # ltrb_offset_target_weight 是 [B, 2, H, W], 我们只需要一个mask
        # 为了方便，可以只用第一个通道作为mask，并扩展
        mask = ltrb_offset_target_weight[:, :1, :, :].expand(-1, 4, -1, -1) > 0

        # 如果你使用的损失函数（如GIoULoss）需要bbox格式，你需要先转换
        # 如果是L1Loss，则可以直接计算
        if self.loss_ltrb.__class__.__name__ in ['GIoULoss', 'IoULoss', 'DIoULoss']:
            # 需要将 ltrb 预测和目标转换回 bbox 格式
            # 这一步比较复杂，需要从 heatmap 的坐标反算
            # 简单起见，我们这里假设使用 L1Loss，如果用IoU Loss需要额外写一个转换函数
            # 这里先用 L1Loss 作为示例
            loss_ltrb = self.loss_ltrb(
                ltrb_pred[mask],
                ltrb_target[mask],
                avg_factor=avg_factor * 4)  # *2或*4取决于如何定义avg_factor
        else:  # L1Loss, SmoothL1Loss
            loss_ltrb = self.loss_ltrb(
                ltrb_pred,
                ltrb_target,
                weight=mask.float(),  # 使用mask作为权重
                avg_factor=avg_factor * 2)

        # offset loss 计算不变
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            ltrb_offset_target_weight,  # 权重可以复用
            avg_factor=avg_factor * 2)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            # 【修改14】返回新的loss
            loss_ltrb=loss_ltrb,
            loss_offset=loss_offset)

    def get_targets(self, gt_bboxes: List[Tensor], gt_labels: List[Tensor],
                    feat_shape: tuple, img_shape: tuple) -> Tuple[dict, int]:
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (tuple): feature map shape with value [B, _, H, W]
            img_shape (tuple): image shape.

        Returns:
            tuple[dict, float]: The float value is mean avg_factor, the dict
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - ltrb_target (Tensor): targets of ltrb predict, shape \
                   (B, 4, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        # 同样，使用 new_zeros 来确保 device 和 dtype 正确，初始化heatmap，ltrb_target，offset_target，ltrb_offset_target_weight
        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        # 【修改6】创建 ltrb_target 替代 wh_target，注意通道数为 4
        ltrb_target = gt_bboxes[-1].new_zeros([bs, 4, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        # 这个权重可以复用，因为 ltrb 和 offset 的 target 都在同一个位置
        ltrb_offset_target_weight = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])  # 形状是(B, 2, H, W)，下面会扩展

        for batch_id in range(bs):

            gt_bbox = gt_bboxes[batch_id]
            if gt_bbox.size(0) == 0:  # 处理没有目标的图片
                continue

            gt_label = gt_labels[batch_id]
            scaled_center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            scaled_center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2

            scaled_centers = torch.cat((scaled_center_x, scaled_center_y), dim=1)

            for j, ct in enumerate(scaled_centers): # 遍历一张图片中所有gt
                ctx_int, cty_int = ct.int()
                ctx, cty = ct

                # 获取原始gt_bbox的坐标
                x1, y1, x2, y2 = gt_bbox[j]

                # 计算 l, r, t, b 目标
                # l = 中心点x - 左边框x
                # r = 右边框x - 中心点x
                # t = 中心点y - 上边框y
                # b = 下边框y - 中心点y
                # 注意：ct是heatmap尺度下的中心点，bbox是原图尺度，需要统一
                # 所以我们反过来，在原图尺度下计算l,r,t,b，再缩放到heatmap尺度

                center_x_orig = (x1 + x2) / 2
                center_y_orig = (y1 + y2) / 2

                l = (center_x_orig - x1) * width_ratio
                r = (x2 - center_x_orig) * width_ratio
                t = (center_y_orig - y1) * height_ratio
                b = (y2 - center_y_orig) * height_ratio

                # 将 l,r,t,b 存入 target  特征图尺度下的坐标
                ltrb_target[batch_id, 0, cty_int, ctx_int] = l
                ltrb_target[batch_id, 1, cty_int, ctx_int] = t
                ltrb_target[batch_id, 2, cty_int, ctx_int] = r
                ltrb_target[batch_id, 3, cty_int, ctx_int] = b

                # heatmap 和 offset 的目标生成逻辑不变
                scale_box_h = (y2 - y1) * height_ratio
                scale_box_w = (x2 - x1) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j] # 获取类别编号
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                # 【修改8】权重目标现在用于 ltrb 和 offset
                # 原来的权重是 [B, 2, H, W]，对于4通道的ltrb不够用
                # 我们只需要一个 [B, 1, H, W] 的mask即可，在loss计算时广播
                ltrb_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum()) # 整个batch的gt个数，用来计算loss，loss/avg_factor

        # 【修改9】更新返回的字典
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            ltrb_target=ltrb_target,
            offset_target=offset_target,
            ltrb_offset_target_weight=ltrb_offset_target_weight)
        return target_result, avg_factor

    def predict_by_feat(self,
                        center_heatmap_preds: List[Tensor],
                        ltrb_preds: List[Tensor],
                        offset_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = True,
                        with_nms: bool = False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            ltrb_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 4, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            batch_img_metas (list[dict], optional): Batch image meta info.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(center_heatmap_preds) == len(ltrb_preds) == len(
            offset_preds) == 1
        result_list = []
        for img_id in range(len(batch_img_metas)):
            result_list.append(
                self._predict_by_feat_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    ltrb_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    batch_img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _predict_by_feat_single(self,
                                center_heatmap_pred: Tensor,
                                ltrb_preds: Tensor,
                                offset_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True,
                                with_nms: bool = False) -> InstanceData:
        """Transform outputs of a single image into bbox results.
        # 将模型输出解码得到box坐标+-
        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            ltrb_pred (Tensor): ltrb heatmap for current level with shape
                (1, 4, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        batch_det_bboxes, batch_labels = self._decode_heatmap(
            center_heatmap_pred,
            ltrb_preds,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
        [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border

        if rescale and 'scale_factor' in img_meta:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        results = InstanceData()
        results.bboxes = det_bboxes[..., :4]
        results.scores = det_bboxes[..., 4]
        results.labels = det_labels
        return results

    # 在 CenterNetHead 的 _decode_heatmap 方法中
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
        tl_x = (topk_xs - ltrb[..., 0]) * radio_w
        tl_y = (topk_ys - ltrb[..., 1]) * radio_h
        br_x = (topk_xs + ltrb[..., 2]) * radio_w
        br_y = (topk_ys + ltrb[..., 3]) * radio_h

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes: Tensor, labels: Tensor,
                    cfg: ConfigDict) -> Tuple[Tensor, Tensor]:
        """bboxes nms."""
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels