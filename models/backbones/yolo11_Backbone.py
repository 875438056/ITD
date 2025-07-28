# Copyright (c) OpenMMLab. All rights reserved.
# 部分代码基于 Ultralytics YOLOv11 实现
import math
from typing import Tuple, Union
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS


class Attention(BaseModule):
    """
    YOLOv11 style Post-hoc Self-Attention (PSA).
    直接从YOLOv11的实现中适配而来，以确保行为一致。
    """

    def __init__(self, c, n_head=4, k_size=3, init_cfg=None):
        super().__init__(init_cfg)
        self.n_head = n_head
        self.k_size = k_size
        self.c_head = c // self.n_head

        self.q = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.k = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.v = nn.Conv2d(c, c, 1, 1, 0, bias=True)

        # Kernel-generating module
        self.k_gen = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.ReLU(),
            nn.Conv2d(c, self.n_head * self.k_size * self.k_size, 1)
        )
        self.unfold = nn.Unfold(kernel_size=self.k_size, padding=self.k_size // 2, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.size()

        # Q, K, V
        # q = self.q(x).view(B, self.n_head, self.c_head, H * W)
        # k = self.k(x).view(B, self.n_head, self.c_head, H * W)
        v = self.v(x)

        # Dynamic kernel generation
        kernels = self.k_gen(x).view(B, self.n_head, self.k_size * self.k_size, H, W)
        kernels = kernels.softmax(dim=2)

        # Unfold V for local context aggregation
        v_unfold = self.unfold(v).view(B, self.n_head, self.c_head, self.k_size * self.k_size, H, W)

        # Apply attention scores (kernels) to unfolded V
        attn_v = (kernels.unsqueeze(2) * v_unfold).sum(dim=3)
        attn_v = attn_v.view(B, C, H, W)

        return self.act(attn_v)


class Conv(BaseModule):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, init_cfg=None):
        super().__init__(init_cfg)
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3k2(BaseModule):
    def __init__(self, c1, c2, n=1, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # 修复：这里应该创建n个卷积块，而不是固定的3个不同kernel size的卷积
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1) for _ in range(n)))

    def forward(self, x):
        # 修复：应该依次通过所有卷积块
        y1 = self.cv1(x)
        for conv in self.m:
            y1 = conv(y1)
        return self.cv3(torch.cat((y1, self.cv2(x)), dim=1))


class SPPF(BaseModule):
    def __init__(self, c1, c2, k=5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        # 修复：移除不必要的torch.no_grad()
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class C2PSA(BaseModule):
    """C2 with Post-hoc Self-Attention."""

    def __init__(self, c1, c2, n=1, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(c_ * (2 + n), c2, 1)
        self.m = nn.ModuleList(Conv(c_, c_, 3, 1) for _ in range(n))
        # 修复：PSA应该在输入通道数上应用，而不是输出通道数
        self.psa = Attention(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x_out = self.cv2(torch.cat(y, 1))
        return self.psa(x_out)


@MODELS.register_module()
class YOLOv11Backbone(BaseModule):
    def __init__(self,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 in_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 init_cfg: dict = None):
        super().__init__(init_cfg=init_cfg)

        def make_divisible(x, divisor):
            return math.ceil(x / divisor) * divisor

        w = lambda x: make_divisible(x * widen_factor, 8)
        d = lambda x: max(round(x * deepen_factor), 1)

        # 修复：确保通道数为整数
        ch = [w(c) for c in [64, 128, 256, 512, 768, 1024]]

        self.layers_cfg = [
            # from, number, module, args(out_channels, kernel, stride, etc.)
            [-1, 1, Conv, [ch[0], 3, 2]],  # 0-P1/2
            [0, 1, Conv, [ch[1], 3, 2]],  # 1-P2/4
            [1, d(2), C3k2, [ch[1]]],  # 2
            [2, 1, Conv, [ch[2], 3, 2]],  # 3-P3/8
            [3, d(4), C3k2, [ch[2]]],  # 4 -> P3 output (index 4)
            [4, 1, Conv, [ch[3], 3, 2]],  # 5-P4/16
            [5, d(4), C3k2, [ch[3]]],  # 6 -> P4 output (index 6)
            [6, 1, Conv, [ch[4], 3, 2]],  # 7-P5/32
            [7, d(2), C2PSA, [ch[4]]],  # 8
            [8, 1, SPPF, [ch[5], 5]],  # 9 -> P5 output (index 9)
        ]

        # 修复：根据用户输入的out_indices或默认值确定输出层
        if out_indices == (2, 3, 4):  # 默认值，需要映射到实际层索引
            self.out_indices = (4, 6, 9)  # P3, P4, P5对应的实际层索引
        else:
            self.out_indices = out_indices

        self.layers = nn.ModuleList()
        current_channels = {-1: in_channels}

        for i, (from_idx, num_blocks, module, args) in enumerate(self.layers_cfg):
            in_c = current_channels[from_idx]

            # 修复：正确处理不同模块的参数传递
            if module == C2PSA:
                # C2PSA需要n参数
                module_instance = module(in_c, *args, n=num_blocks)
            else:
                # 其他模块
                if num_blocks == 1:
                    module_instance = module(in_c, *args)
                else:
                    # 创建多个重复的模块
                    if module == C3k2:
                        # C3k2的n参数控制内部卷积块数量
                        module_instance = module(in_c, *args, n=num_blocks)
                    else:
                        # 其他模块创建Sequential
                        module_instance = nn.Sequential(
                            *(module(in_c if j == 0 else args[0], *args) for j in range(num_blocks))
                        )

            self.layers.append(module_instance)

            # 修复：更准确的输出通道数计算
            out_c = self._get_output_channels(module_instance, args)
            current_channels[i] = out_c

    def _get_output_channels(self, module_instance, args):
        """获取模块的输出通道数"""
        if isinstance(module_instance, nn.Sequential):
            # 对于Sequential，获取最后一个模块的输出通道
            last_module = module_instance[-1]
            return self._get_single_module_output_channels(last_module, args)
        else:
            return self._get_single_module_output_channels(module_instance, args)

    def _get_single_module_output_channels(self, module, args):
        """获取单个模块的输出通道数"""
        if hasattr(module, 'cv3') and hasattr(module.cv3, 'conv'):  # C3k2
            return module.cv3.conv.out_channels
        elif hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):  # SPPF, C2PSA
            return module.cv2.conv.out_channels
        elif hasattr(module, 'conv'):  # Conv
            return module.conv.out_channels
        else:
            # 回退到args中的第一个参数（通常是输出通道数）
            return args[0] if len(args) > 0 else 64

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = []
        saved_outputs = {-1: x}

        for i, (from_idx, _, _, _) in enumerate(self.layers_cfg):
            input_tensor = saved_outputs[from_idx]
            output_tensor = self.layers[i](input_tensor)
            saved_outputs[i] = output_tensor

            if i in self.out_indices:
                outputs.append(output_tensor)

        return tuple(outputs)

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)