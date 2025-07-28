# 文件路径: F:/ITD/models/heads/custom_centernet_head.py
import torch
from typing import List, Tuple
from torch import Tensor

# 假设已正确导入 SampleList, InstanceData 等 MMDetection 类型
# from mmdet.structures import SampleList
# from mmdet.structures.instance_data import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads import CenterNetHead
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.misc import unpack_gt_instances


@MODELS.register_module()
class CentroidNetHead(CenterNetHead):
    """
    完全符合 MMDetection 3.x 数据流的自定义 CenterNetHead
    """

    def get_targets(self,
                    batch_data_samples: List['DetDataSample'],
                    feat_shape: Tuple[int, ...]) -> Tuple[dict, float]:
        """
        get_targets 方法接收包含所有信息的 batch_data_samples，
        并在此函数内部完成所有数据的解包。
        """
        # 从 data_sample 中解包需要的数据
        batch_gt_instances = [
            data_sample.gt_instances for data_sample in batch_data_samples
        ]
        # 从 data_sample 中获取 img_shape
        # img_h, img_w = batch_data_samples[0].img_shape

        # 从 gt_instances 中获取 bbox, label 和自定义的 centroids
        gt_bboxes = [gt_inst.bboxes for gt_inst in batch_gt_instances]
        gt_labels = [gt_inst.labels for gt_inst in batch_gt_instances]
        gt_centroids = [gt_inst.gt_centroids for gt_inst in batch_gt_instances]

        bs, _, feat_h, feat_w = feat_shape

        # width_ratio = float(feat_w / img_w)
        # height_ratio = float(feat_h / img_h)

        device = gt_bboxes[0].device
        center_heatmap_target = torch.zeros(
            [bs, self.num_classes, feat_h, feat_w], device=device)
        wh_target = torch.zeros([bs, 2, feat_h, feat_w], device=device)
        offset_target = torch.zeros([bs, 2, feat_h, feat_w], device=device)
        wh_offset_target_weight = torch.zeros(
            [bs, 2, feat_h, feat_w], device=device)

        for batch_id in range(bs):
            data_sample = batch_data_samples[batch_id]
            gt_instances = data_sample.gt_instances
            img_h, img_w = data_sample.img_shape

            width_ratio = float(feat_w / img_w)
            height_ratio = float(feat_h / img_h)

            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            gt_center_numpy_array = gt_centroids[batch_id]
            gt_center_tensor = torch.from_numpy(gt_center_numpy_array).to(device)# numpy->tensor

            scale_gt_centers = gt_center_tensor * torch.tensor(
                [width_ratio, height_ratio], device=device)

            for j, ct in enumerate(scale_gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius((scale_box_h, scale_box_w), min_overlap=0.7)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def loss(self, x: Tuple[Tensor],
             batch_data_samples: List['DetDataSample']) -> dict:
        """
        loss 方法将收到的 batch_data_samples 原封不动地传递给 get_targets。
        """
        outs = self(x)
        center_heatmap_pred_list, wh_pred_list, offset_pred_list = outs
        center_heatmap_pred = center_heatmap_pred_list[0]
        wh_pred = wh_pred_list[0]
        offset_pred = offset_pred_list[0]

        feat_shape = center_heatmap_pred.shape

        # 【【【 正确的调用方式 】】】
        # 直接传递 batch_data_samples，不进行任何预处理
        target_result, avg_factor = self.get_targets(batch_data_samples, feat_shape)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)