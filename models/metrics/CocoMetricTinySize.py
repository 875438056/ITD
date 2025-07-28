# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from mmdet.evaluation.functional import eval_recalls


@METRICS.register_module()
class CocoTinyMetric(BaseMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    (其余文档字符串保持不变)

    Args:
        ... (省略)
        area_Rng (list, optional): Custom area ranges for evaluation.
            Each element is a list of [min_area, max_area].
            Defaults to various small object sizes.
        area_RngLbl (list[str], optional): Labels for the custom area ranges.
            The length must match `area_Rng`. Defaults to labels
            corresponding to the default `area_Rng`.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False,
                 # ===== 您自定义的面积参数 =====
                 area_Rng: Optional[list] = None,
                 area_RngLbl: Optional[list[str]] = None,) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # ===== 处理自定义面积参数 =====
        # 如果用户未提供，则使用COCO默认值
        if area_Rng is None or area_RngLbl is None:
            self.area_Rng = [
                [0 ** 2, 1e5 ** 2],  # all
                [0 ** 2, 32 ** 2],  # small
                [32 ** 2, 96 ** 2],  # medium
                [96 ** 2, 1e5 ** 2],  # large
            ]
            self.area_RngLbl = ['all', 'small', 'medium', 'large']
        else:
            # 校验自定义参数
            if len(area_Rng) != len(area_RngLbl):
                raise ValueError(
                    'Length of `area_Rng` and `area_RngLbl` must be the same.')
            # COCOeval期望第一个范围是 'all'
            if area_RngLbl[0] != 'all' or area_Rng[0] != [0, 1e5 ** 2]:
                # 自动为用户添加 'all' 范围
                self.area_Rng = [[0, 1e5 ** 2]] + list(area_Rng)
                self.area_RngLbl = ['all'] + list(area_RngLbl)
            else:
                self.area_Rng = area_Rng
                self.area_RngLbl = area_RngLbl

        self.classwise = classwise
        self.use_mp_eval = use_mp_eval
        self.proposal_nums = list(proposal_nums)

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead.')

        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        self.cat_ids = None
        self.img_ids = None

    # ===== fast_eval_recall, xyxy2xywh, results2json, gt_to_coco_json, process 方法保持不变 =====
    # ... (此处省略未作修改的几个方法，请保留您原有的实现) ...
    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall."""
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style."""
        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file."""
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file."""
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                       1,
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions."""
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            # if 'bboxes' not in pred or len(pred) == 0:
            #     # 如果当前图片没有检测结果，我们可以直接跳过
            #     # 或者按照 COCO 的标准，添加一个空的记录
            #     # MMDetection 的标准做法是收集所有图片的信息，所以我们继续
            #     pass  # 允许 pred 为空，后续的 results2json 会处理
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                    pred['masks'], torch.Tensor) else pred['masks']
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        # # ==================== 新的统计代码（适用于提供了 ann_file 的情况）====================
        # if self._coco_api:
        #     logger.info('---------- 全验证集 Ground Truth Box 面积统计 (来自 ann_file) ----------')
        #     from collections import Counter
        #     # self._coco_api.anns 是一个包含所有标注的字典，格式为 {ann_id: annotation}
        #     # 我们直接从这里提取面积
        #     # 您的 val.json 中已经包含了 "area" 字段，所以直接使用 ann['area']
        #     all_gt_areas = [ann['area'] for ann in self._coco_api.anns.values() if not ann.get('iscrowd', 0)]
        #
        #     if all_gt_areas:
        #         area_counts = Counter(all_gt_areas)
        #         # 按照面积大小进行排序
        #         sorted_areas = sorted(area_counts.items())
        #         for area, count in sorted_areas:
        #             logger.info(f'  面积为 {int(area):>4} 的Box数量: {count} 个')
        #     else:
        #         logger.info('  在 ann_file 中未找到任何 Ground Truth Box。')
        #     logger.info('-----------------------------------------------------------------')
        # # ================================== 统计代码结束 ==================================

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        result_files = self.results2json(preds, outfile_prefix)
        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    for x in predictions:
                        x.pop('bbox', None)
                coco_dt = self._coco_api.loadRes(predictions)
            except IndexError:
                logger.error('The testing results of the whole dataset is empty.')
                break

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # ===== MODIFICATION START =====
            # 1. 使用您自定义的面积范围和标签
            # areaRng 和 areaRngLbl 的第一个元素应始终是 'all' 和对应的范围
            # self.area_Rng 和 self.area_RngLbl 已在 __init__ 中处理
            coco_eval.params.areaRng = self.area_Rng
            coco_eval.params.areaRngLbl = self.area_RngLbl

            # 2. 动态生成 coco_metric_names 字典
            # The output of coco_eval.stats will be a a 1D array where the
            # order of metrics is determined by the length of areaRngLbl.
            # AP stats come first, then AR stats.
            # e.g. with 4 area labels: [mAP, mAP_50, mAP_75, mAP_area1, mAP_area2, mAP_area3,
            #                           AR@100, AR@300, AR@1000, AR_area1, AR_area2, AR_area3]
            # (Note: COCO API includes 'all' in area labels, so custom labels are N-1)

            # 自定义标签（除'all'之外）
            custom_area_labels = self.area_RngLbl[1:]  # Exclude 'all'

            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2
            }
            # 动态添加自定义面积的 mAP 指标
            for i, label in enumerate(custom_area_labels):
                coco_metric_names[f'mAP_{label}'] = 4 + i

            # AR 指标的起始索引
            ar_start_index = 3 + len(custom_area_labels)
            coco_metric_names.update({
                'AR@100': ar_start_index,
                'AR@300': ar_start_index + 1,
                'AR@1000': ar_start_index + 2
            })
            # 动态添加自定义面积的 AR 指标
            for i, label in enumerate(custom_area_labels):
                # 遵循原格式 AR_<label>@<maxDet>，我们使用最大的maxDet
                coco_metric_names[f'AR_{label}@{self.proposal_nums[-1]}'] = ar_start_index + 3 + i

            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported. '
                            f'Available options are: {list(coco_metric_names.keys())}')
            # ===== MODIFICATION END =====

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # ===== MODIFICATION START =====
                # 3. 动态生成默认的 metric_items
                if metric_items is None:
                    metric_items = ['AR@100', 'AR@300', 'AR@1000']
                    for label in custom_area_labels:
                        metric_items.append(f'AR_{label}@{self.proposal_nums[-1]}')
                # ===== MODIFICATION END =====

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:
                    precisions = coco_eval.eval['precision']
                    assert len(self.cat_ids) == precisions.shape[2]
                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        nm = self._coco_api.loadCats(cat_id)[0]

                        # mAP (IoU=0.50:0.95, area=all)
                        precision_all = precisions[:, :, idx, 0, -1]
                        ap = np.mean(precision_all[precision_all > -1]) if precision_all.size else float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # mAP_50, mAP_75
                        for iou_idx in [0, 5]:  # Index for IoU 0.50 and 0.75
                            precision = precisions[iou_idx, :, idx, 0, -1]
                            ap = np.mean(precision[precision > -1]) if precision.size else float('nan')
                            t.append(f'{round(ap, 3)}')

                        # ===== MODIFICATION START =====
                        # 4. 动态处理 class-wise 的面积指标
                        # area a index 0 is for 'all', custom areas start from 1
                        for area_idx in range(1, len(self.area_RngLbl)):
                            precision = precisions[:, :, idx, area_idx, -1]
                            ap = np.mean(precision[precision > -1]) if precision.size else float('nan')
                            t.append(f'{round(ap, 3)}')
                        # ===== MODIFICATION END =====

                        results_per_category.append(tuple(t))

                    # ===== MODIFICATION START =====
                    # 5. 动态生成 class-wise 表格的表头
                    headers = ['category', 'mAP', 'mAP_50', 'mAP_75']
                    for label in custom_area_labels:
                        headers.append(f'mAP_{label}')
                    # ===== MODIFICATION END =====

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                # ===== MODIFICATION START =====
                # 6. 动态生成默认的 metric_items
                if metric_items is None:
                    metric_items = ['mAP', 'mAP_50', 'mAP_75']
                    for label in custom_area_labels:
                        metric_items.append(f'mAP_{label}')
                # ===== MODIFICATION END =====

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                # ===== MODIFICATION START =====
                # 7. 动态生成 mAP_copypaste 日志
                num_ap_metrics = 3 + len(custom_area_labels)
                ap = coco_eval.stats[:num_ap_metrics]
                ap_strings = [f'{v:.3f}' for v in ap]
                logger.info(f'{metric}_mAP_copypaste: ' + ' '.join(ap_strings))
                # ===== MODIFICATION END =====

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results