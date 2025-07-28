import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.task_modules import (build_assigner, build_sampler)
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmdet.models.dense_heads.anchor_head import anchor_inside_flags
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.registry import MODELS
from mmengine.model import BaseModule
import numpy as np
from mmdet.models.layers.bbox_nms import multiclass_nms


@MODELS.register_module()
class YOLOv11S2Head(BaseDenseHead):
    """YOLOv11 Detection Head.

    Args:
        num_classes (int): Number of categories excluding the background category.
        in_channels (list): Number of channels in the input feature maps.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes. Default: False.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_dfl (dict): Config of Distribution Focal Loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU', inplace=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=0.5),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='ciou',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=7.5),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_dfl=dict(
                     type='DistributionFocalLoss',
                     reduction='sum',
                     loss_weight=1.5),
                 reg_max=16,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(YOLOv11S2Head, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels  # Should be list [C1, C2, C3, C4] for P1,P2,P3,P4
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_obj = MODELS.build(loss_obj)
        self.loss_dfl = MODELS.build(loss_dfl)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Build the head layers
        self._init_layers()

        # DFL conv for regression
        self.dfl_conv = nn.Conv2d(self.reg_max, 1, 1, bias=False).requires_grad_(False)
        self.dfl_conv.weight.data[:] = nn.Parameter(
            torch.arange(self.reg_max, dtype=torch.float).view(1, self.reg_max, 1, 1) / self.reg_max)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        # Create separate conv layers for each FPN level
        for level_idx in range(len(self.strides)):
            cls_convs_per_level = nn.ModuleList()
            reg_convs_per_level = nn.ModuleList()

            for i in range(self.stacked_convs):
                # For first conv, use input channels specific to each level
                if i == 0:
                    if isinstance(self.in_channels, list):
                        chn = self.in_channels[level_idx]
                    else:
                        chn = self.in_channels
                else:
                    chn = self.feat_channels

                if self.use_depthwise:
                    cls_convs_per_level.append(
                        DepthwiseSeparableConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            bias=self.conv_bias))
                    reg_convs_per_level.append(
                        DepthwiseSeparableConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            bias=self.conv_bias))
                else:
                    cls_convs_per_level.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            bias=self.conv_bias))
                    reg_convs_per_level.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            bias=self.conv_bias))

            self.cls_convs.append(cls_convs_per_level)
            self.reg_convs.append(reg_convs_per_level)

        # Prediction layers - separate for each level to handle different channel dimensions
        self.conv_cls = nn.ModuleList([
            nn.Conv2d(self.feat_channels, self.num_classes, 1, bias=self.conv_bias)
            for _ in range(len(self.strides))
        ])
        self.conv_reg = nn.ModuleList([
            nn.Conv2d(self.feat_channels, 4 * self.reg_max, 1, bias=self.conv_bias)
            for _ in range(len(self.strides))
        ])
        self.conv_obj = nn.ModuleList([
            nn.Conv2d(self.feat_channels, 1, 1, bias=self.conv_bias)
            for _ in range(len(self.strides))
        ])

    def forward_single(self, x, level_idx):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Feature map of single level.
            level_idx (int): Index of the FPN level.
        """
        cls_feat = x
        reg_feat = x

        # Classification branch
        for cls_conv in self.cls_convs[level_idx]:
            cls_feat = cls_conv(cls_feat)
        cls_score = self.conv_cls[level_idx](cls_feat)

        # Regression branch
        for reg_conv in self.reg_convs[level_idx]:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = self.conv_reg[level_idx](reg_feat)
        objectness = self.conv_obj[level_idx](reg_feat)

        return cls_score, bbox_pred, objectness

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor. Should contain P1, P2, P3, P4 features.
        Returns:
            tuple: A tuple of classification scores, bbox predictions and
                objectness predictions.
        """
        assert len(feats) == len(self.strides), \
            f"Expected {len(self.strides)} feature levels, but got {len(feats)}"

        results = []
        for i, feat in enumerate(feats):
            result = self.forward_single(feat, i)
            results.append(result)

        # Transpose results to match original format
        cls_scores, bbox_preds, objectnesses = zip(*results)
        return cls_scores, bbox_preds, objectnesses

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        multi_level_anchors = []

        for i, featmap_size in enumerate(featmap_sizes):
            anchors = self.get_single_level_anchors(
                featmap_size, i, device=device)
            multi_level_anchors.append(anchors)

        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = [
            self.get_valid_flags(featmap_sizes, img_metas[i], device)
            for i in range(num_imgs)
        ]

        return anchor_list, valid_flag_list

    def get_single_level_anchors(self, featmap_size, level_idx, device='cuda'):
        """Generate anchors of a single level."""
        h, w = featmap_size
        stride = self.strides[level_idx]

        # Generate grid points
        shift_x = torch.arange(0, w, device=device) * stride
        shift_y = torch.arange(0, h, device=device) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shifts = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)

        return shifts

    def get_valid_flags(self, featmap_sizes, img_meta, device='cuda'):
        """Get valid flags of anchors in each feature level."""
        multi_level_flags = []
        for i, featmap_size in enumerate(featmap_sizes):
            h, w = featmap_size
            valid_feat_h = min(int(np.ceil(img_meta['img_shape'][0] / self.strides[i])), h)
            valid_feat_w = min(int(np.ceil(img_meta['img_shape'][1] / self.strides[i])), w)
            flags = self.get_single_level_flags(
                (h, w), (valid_feat_h, valid_feat_w), device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def get_single_level_flags(self, featmap_size, valid_size, device='cuda'):
        """Get valid flags of a single feature level."""
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = torch.meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def loss_single(self, cls_score, bbox_pred, objectness, anchors, labels,
                    label_weights, bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level."""
        # Classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # Regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * self.reg_max)

        # DFL loss
        if self.loss_dfl is not None:
            # Convert bbox_pred to distribution format
            bbox_pred_corners = bbox_pred.reshape(-1, 4, self.reg_max)
            bbox_targets_corners = bbox_targets

            loss_dfl = self.loss_dfl(
                bbox_pred_corners.reshape(-1, self.reg_max),
                bbox_targets_corners.reshape(-1),
                weight=bbox_weights.reshape(-1),
                avg_factor=num_total_samples)
        else:
            loss_dfl = bbox_pred.sum() * 0

        # Bbox loss (IoU loss)
        if bbox_weights.sum() > 0:
            # Decode bbox predictions
            bbox_pred_decoded = self.bbox_coder.decode(anchors, bbox_pred)
            loss_bbox = self.loss_bbox(
                bbox_pred_decoded,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0

        # Objectness loss
        objectness = objectness.permute(0, 2, 3, 1).reshape(-1)
        obj_targets = (labels > 0).float()
        loss_obj = self.loss_obj(
            objectness, obj_targets, label_weights, avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_obj, loss_dfl

    @property
    def strides(self):
        """Get strides of different FPN levels."""
        return [2, 4, 8, 16]  # Modified for P1, P2, P3, P4 features

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list,
                    img_metas, gt_bboxes_ignore_list=None,
                    gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors."""
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # Anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # Compute targets for each image
        all_labels, all_label_weights, all_bbox_targets, all_bbox_weights = multi_apply(
            self._get_targets_single,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        # No valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # Sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in all_label_weights])
        num_total_neg = sum([max(inds.numel(), 1) for inds in all_label_weights])

        # Split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes,
                            gt_bboxes_ignore, gt_labels, img_meta,
                            label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for a single image."""
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7

        # Assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # Map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_by_feat(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # Anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # Concat all level anchors to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox, losses_obj, losses_dfl = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            objectnesses,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_obj=losses_obj,
            loss_dfl=losses_dfl)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions."""
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            obj_pred_list = [
                objectnesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                obj_pred_list, mlvl_anchors,
                                                img_shape, scale_factor, cfg,
                                                rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           objectnesses,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions."""
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_confids = []

        for cls_score, bbox_pred, objectness, anchors in zip(
                cls_scores, bbox_preds, objectnesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4 * self.reg_max)
            objectness = objectness.permute(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * objectness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                objectness = objectness[topk_inds]

            # Decode bboxes
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_confids.append(objectness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_confids = torch.cat(mlvl_confids)

        if self.use_sigmoid_cls:
            # Add background class for consistency
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores * mlvl_confids[:, None],
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores * mlvl_confids[:, None]