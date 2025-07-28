# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from multiprocessing import Pool
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import ProgressBar, check_file_exist, mkdir_or_exist

from mmdet.datasets import get_loading_pipeline
from mmdet.evaluation import eval_map
from mmdet.registry import DATASETS, RUNNERS
from mmdet.structures import DetDataSample
from mmdet.utils import replace_cfg_vals, update_data_root
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon

from mmdet.visualization.palette import _get_adaptive_scales, get_palette, jitter_color

# 我们定义一个自己的 Visualizer，它继承自官方的 DetLocalVisualizer
class RightAlignVisualizer(DetLocalVisualizer):

    # 我们只需要重写 _draw_instances 这一个方法
    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """这个方法大部分内容是从源码复制的，只修改了一行"""
        self.set_image(image)

        if 'bboxes' in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)


            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            # 直接遍历每个 bbox，从而获取它的所有坐标 (x1, y1, x2, y2)
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                # 根据您的要求，将文本定位在检测框右上角的外侧
                # bbox[2] 是右上角的 x 坐标 (x2)
                # bbox[1] 是右上角的 y 坐标 (y1)
                # 我们将文本的起始点设置在 (x2 + 5, y1)，+5 是为了留出一点空隙
                text_position = np.array([bbox[2]+2, bbox[1]]) #(x,y)

                # 创建标签文本
                if 'label_names' in instances:
                    label_text = instances.label_names[i]
                else:
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                # 使用我们手动计算的新坐标来绘制文本
                self.draw_texts(
                    label_text,
                    text_position,  # <-- 使用数组作为坐标，这是绘图函数接受的格式
                    colors=text_colors[i],
                    font_sizes=int(6),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.1,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        # --- 以下的 mask 处理部分和源码完全一样，保持不变 ---
        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()
            masks = masks.astype(bool)
            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]
            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)
            if len(labels) > 0 and \
                    ('bboxes' not in instances or
                     instances.bboxes.sum() == 0):
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)
                for i, (pos, label) in enumerate(zip(positions, labels)):
                    if 'label_names' in instances:
                        label_text = instances.label_names[i]
                    else:
                        label_text = classes[
                            label] if classes is not None else f'class {label}'
                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f': {score}'
                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])
        return self.get_image()


def bbox_map_eval(det_result, annotation, nproc=4):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

        nproc (int): Processes used for computing mAP.
            Default: 4.

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]
    # mAP
    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    processes = []
    workers = Pool(processes=nproc)
    for thr in iou_thrs:
        p = workers.apply_async(eval_map, (bbox_det_result, [annotation]), {
            'iou_thr': thr,
            'logger': 'silent',
            'nproc': 1
        })
        processes.append(p)

    workers.close()
    workers.join()

    mean_aps = []
    for p in processes:
        mean_aps.append(p.get()[0])

    return sum(mean_aps) / len(mean_aps)


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True.
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0.
        runner (:obj:`Runner`): The runner of the visualization process.
    """

    def __init__(self, show=False, wait_time=0, score_thr=0, runner=None):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr

        # self.visualizer = DetLocalVisualizer()
        self.visualizer = RightAlignVisualizer(line_width=0.5)
        self.runner = runner
        self.evaluator = runner.test_evaluator

    def _save_image_gts_results(self,
                                dataset,
                                results,
                                performances,
                                out_dir=None,
                                task='det'):
        """Display or save image with groung truths and predictions from a
        model.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            performances (dict): A dict contains samples's indices
                in dataset and model's performance on them.
            out_dir (str, optional): The filename to write the image.
                Defaults: None.
            task (str): The task to be performed. Defaults: 'det'
        """
        # --- 添加下面这行测试代码 ---
        print(f"[DEBUG] Now in _save_image_gts_results. Preparing to save {len(performances)} images.")

        mkdir_or_exist(out_dir)

        for performance_info in performances:
            index, performance = performance_info
            data_info = dataset[index]
            data_info['gt_instances'] = data_info['instances']

            # calc save file path
            filename = data_info['img_path']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(performance, 3)) + name
            out_file = osp.join(out_dir, save_filename)

            if task == 'det':
                gt_instances = InstanceData()
                gt_instances.bboxes = [
                    d['bbox'] for d in data_info['gt_instances']
                ]
                gt_instances.labels = [
                    d['bbox_label'] for d in data_info['gt_instances']
                ]

                pred_instances = InstanceData()
                pred_instances.bboxes = results[index]['pred_instances'][
                    'bboxes']
                pred_instances.labels = results[index]['pred_instances'][
                    'labels']
                pred_instances.scores = results[index]['pred_instances'][
                    'scores']

                data_samples = DetDataSample()
                data_samples.pred_instances = pred_instances
                data_samples.gt_instances = gt_instances

            elif task == 'seg':
                gt_panoptic_seg = PixelData()
                gt_panoptic_seg.sem_seg = [
                    d['gt_seg_map'] for d in data_info['gt_instances']
                ]

                pred_panoptic_seg = PixelData()
                pred_panoptic_seg.sem_seg = results[index][
                    'pred_panoptic_seg']['sem_seg']

                data_samples = DetDataSample()
                data_samples.pred_panoptic_seg = pred_panoptic_seg
                data_samples.gt_panoptic_seg = gt_panoptic_seg

            # --- 添加下面这行代码 ---
            print(f"    [DEBUG] Reading original image from: {filename}")

            img = mmcv.imread(filename, channel_order='rgb')

            # --- 添加下面这行代码 ---
            print(f"    [DEBUG] Image read successfully. Now calling visualizer...")


            self.visualizer.add_datasample(
                'image',
                img,
                data_samples,
                show=self.show,
                draw_gt=False,
                pred_score_thr=self.score_thr,
                out_file=out_file)

            # --- 添加下面这行代码 ---
            print(f"    [DEBUG] Visualizer finished for image index {index}.")

    def evaluate_and_show(self,
                          dataset,
                          results,
                          topk=20,
                          show_dir='work_dir'):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
        """

        self.visualizer.dataset_meta = dataset.metainfo

        assert topk > 0
        if (topk * 2) > len(dataset):
            topk = len(dataset) // 2

        good_dir = osp.abspath(osp.join(show_dir, 'good'))
        bad_dir = osp.abspath(osp.join(show_dir, 'bad'))

        if 'pred_panoptic_seg' in results[0].keys():
            good_samples, bad_samples = self.panoptic_evaluate(
                dataset, results, topk=topk)
            self._save_image_gts_results(
                dataset, results, good_samples, good_dir, task='seg')
            self._save_image_gts_results(
                dataset, results, bad_samples, bad_dir, task='seg')
        elif 'pred_instances' in results[0].keys():
            good_samples, bad_samples = self.detection_evaluate(
                dataset, results, topk=topk)
            # --- 添加下面这两行代码 ---
            print(f"\n[DEBUG] Found {len(good_samples)} good samples to show.")
            print(f"[DEBUG] Found {len(bad_samples)} bad samples to show.\n")

            self._save_image_gts_results(
                dataset, results, good_samples, good_dir, task='det')
            self._save_image_gts_results(
                dataset, results, bad_samples, bad_dir, task='det')
        else:
            raise 'expect \'pred_panoptic_seg\' or \'pred_instances\' \
                in dict result'

    def detection_evaluate(self, dataset, results, topk=20, eval_fn=None):
        """Evaluation for object detection.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            eval_fn (callable, optional): Eval function, Default: None.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_mAPs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_mAPs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """

        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)

        prog_bar = ProgressBar(len(results))
        _mAPs = {}
        data_info = {}
        for i, (result, ) in enumerate(zip(results)):
            # --- 只处理前 50 张图片用于快速测试 ---
            if i >= 30:
                break
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_data(i)
            data_info['bboxes'] = data_info['gt_bboxes'].tensor
            data_info['labels'] = data_info['gt_bboxes_labels']

            pred = result['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(dataset.metainfo['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)
            mAP = eval_fn(dets, data_info)
            # --- 添加下面这行代码 ---
            print(f"[DEBUG] Image {i}: Calculated mAP = {mAP:.4f}")

            _mAPs[i] = mAP
            prog_bar.update()
        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        good_mAPs = _mAPs[-topk:]
        bad_mAPs = _mAPs[:topk]

        return good_mAPs, bad_mAPs

    def panoptic_evaluate(self, dataset, results, topk=20):
        """Evaluation for panoptic segmentation.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Panoptic segmentation results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_pqs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_pqs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """
        pqs = {}
        prog_bar = ProgressBar(len(results))

        for i in range(len(results)):
            data_sample = {}
            for k in dataset[i].keys():
                data_sample[k] = dataset[i][k]

            for k in results[i].keys():
                data_sample[k] = results[i][k]

            self.evaluator.process([data_sample])
            metrics = self.evaluator.evaluate(1)

            pqs[i] = metrics['coco_panoptic/PQ']
            prog_bar.update()

        # descending select topk image
        pqs = list(sorted(pqs.items(), key=lambda kv: kv[1]))
        good_pqs = pqs[-topk:]
        bad_pqs = pqs[:topk]

        return good_pqs, bad_pqs


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='saved Number of the highest topk '
        'and lowest topk after index sorting')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0,
        help='score threshold (default: 0.)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    cfg.test_dataloader.dataset.test_mode = True

    cfg.test_dataloader.pop('batch_size', 0)
    if cfg.train_dataloader.dataset.type in ('MultiImageMixDataset',
                                             'ClassBalancedDataset',
                                             'RepeatDataset'):
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.dataset.pipeline)
    elif cfg.train_dataloader.dataset.type in ('ConcatDataset', ):
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.datasets[0].pipeline)
    else:
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.pipeline)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    outputs = load(args.prediction_path)

    cfg.work_dir = args.show_dir
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    result_visualizer = ResultVisualizer(args.show, args.wait_time,
                                         args.show_score_thr, runner)
    result_visualizer.evaluate_and_show(
        dataset, outputs, topk=args.topk, show_dir=args.show_dir)


if __name__ == '__main__':
    main()
