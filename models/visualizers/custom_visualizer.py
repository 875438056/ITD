# F:/ITD/models/visualizers/custom_visualizer.py

from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
import torch
from mmengine.visualization.vis_backend import TensorboardVisBackend
import numpy as np


@VISUALIZERS.register_module()
class CustomDetVisualizer(DetLocalVisualizer):
    """
    一个自定义的可视化器，用于精细控制边界框的线条宽度和字体大小。
    """

    def __init__(self,
                 name='visualizer',
                 image=None,
                 vis_backends=None,
                 save_dir=None,
                 line_width=0.5,
                 font_size=8,
                 bbox_color=None,):
        super().__init__(name, image, vis_backends, save_dir)

        self.line_width = line_width
        self.font_size = font_size
        self.bbox_color = bbox_color

    def _draw_instances(self, image, instances, classes, palette):
        """
        重写绘制实例的方法。
        1. 过滤掉中心点在图像外的边界框。
        2. 将剩余的框限制在图像范围内。
        3. 绘制实例。
        """
        self.set_image(image)

        bboxes = instances.get('bboxes', None)
        labels = instances.get('labels', None)

        if bboxes is None or len(bboxes) == 0:
            return self.get_image()

        # 获取图像的高度和宽度
        height, width = image.shape[:2]

        # --- 新增代码：过滤掉中心点在图像外的边界框 ---
        # 计算所有边界框的中心点坐标 (cx, cy)
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2

        # 创建一个布尔掩码 (boolean mask)，保留中心点在图像内的框
        # 条件：0 <= cx < width 并且 0 <= cy < height
        valid_mask = (centers_x >= 0) & (centers_x < width) & \
                     (centers_y >= 0) & (centers_y < height)

        # 应用掩码，只保留有效的边界框和对应的标签
        bboxes = bboxes[valid_mask]
        labels = labels[valid_mask]

        # 如果过滤后没有剩下任何框，则直接返回
        if len(bboxes) == 0:
            return self.get_image()
        # --- 新增过滤代码结束 ---

        # --- 将有效框的边界限制在图像范围内 ---
        bboxes = bboxes.clone()
        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0, max=width - 1)
        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0, max=height - 1)

        # 确保边界框格式正确：x1 <= x2, y1 <= y2
        # 如果clamp后出现x1 > x2或y1 > y2的情况，需要进一步过滤
        width_valid = bboxes[:, 2] > bboxes[:, 0]
        height_valid = bboxes[:, 3] > bboxes[:, 1]
        bbox_valid = width_valid & height_valid

        bboxes = bboxes[bbox_valid]
        labels = labels[bbox_valid]

        # 如果最终过滤后没有剩下任何框，则直接返回
        if len(bboxes) == 0:
            return self.get_image()
        # --- 限制代码结束 ---

        # 使用最终处理过的边界框进行后续绘制
        if self.bbox_color == None:
            _bbox_color = [palette[label] for label in labels]
        else:
            _bbox_color = self.bbox_color

        # 1. 绘制边界框
        self.draw_bboxes(
            bboxes,
            edge_colors='r',
            line_widths=self.line_width)

        # 2. 准备文本
        texts = []
        for i, label in enumerate(labels):
            text = classes[label]
            # 处理scores：从原始instances中根据最终有效的索引获取
            if 'scores' in instances:
                # 需要通过原始索引来获取对应的分数
                original_indices = torch.where(valid_mask)[0]
                final_indices = original_indices[bbox_valid]
                if i < len(final_indices):
                    score = instances.scores[final_indices[i]]
                    # text = f'{text}: {score:.2f}'
                    text = f'{score:.2f}'
            texts.append(text)

        # 3. 计算文本位置
        top_right_corners = torch.stack(
            [bboxes[:, 2]+2, bboxes[:, 1]], dim=-1)

        # 4. 绘制文本
        self.draw_texts(
            texts,
            positions=top_right_corners,
            # vertical_alignments='top',
            # horizontal_alignments='right',
            colors='r',
            font_sizes=self.font_size)

        return self.get_image()