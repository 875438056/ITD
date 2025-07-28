# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.registry import MODELS
from mmengine.model import BaseModule

print("--- DEBUG: Successfully imported YOLOv11 Backbone file! ---")


class C2fPSA(BaseModule):
    """C2f with Position-aware Spatial Attention (PSA) - YOLOv11's innovation"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = False,
                 g: int = 1,
                 e: float = 0.5,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, 2 * self.c, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule((2 + n) * self.c, c2, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.m = nn.ModuleList(PSABottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                               for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through C2fPSA layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class PSABottleneck(BaseModule):
    """Position-aware Spatial Attention Bottleneck"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 shortcut: bool = True,
                 g: int = 1,
                 k: Tuple = (3, 3),
                 e: float = 0.5,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, k[0], 1, k[0][0] // 2, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_, c2, k[1], 1, k[1][0] // 2, groups=g, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.psa = PSA(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with PSA attention."""
        out = self.cv2(self.cv1(x))
        out = self.psa(out)
        return x + out if self.add else out


class PSA(BaseModule):
    """Position-aware Spatial Attention"""

    def __init__(self, c: int, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.c = c
        self.cv1 = nn.Conv2d(c, c, 1, bias=False)
        self.cv2 = nn.Conv2d(c, c, 1, bias=False)
        self.cv3 = nn.Conv2d(c, c, 1, bias=False)
        self.cv4 = nn.Conv2d(c, c // 2, 1, bias=False)
        self.cv5 = nn.Conv2d(c // 2, c, 1, bias=False)
        self.norm = nn.LayerNorm(c)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with position-aware attention."""
        B, C, H, W = x.shape

        # Multi-head attention
        q = self.cv1(x).view(B, C, H * W).transpose(1, 2)
        k = self.cv2(x).view(B, C, H * W)
        v = self.cv3(x).view(B, C, H * W).transpose(1, 2)

        # Attention computation
        attn = (q @ k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).view(B, C, H, W)

        # Feed forward
        out = self.cv5(torch.relu(self.cv4(out)))

        return x + out


class C3k2(BaseModule):
    """C3k2 Block - YOLOv11's efficient block"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 c3k: bool = False,
                 e: float = 0.5,
                 g: int = 1,
                 shortcut: bool = True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, 2 * self.c, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(self.c * (2 + n), c2, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        if c3k:
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                                   for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through C3k2 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(BaseModule):
    """C3k Block for lightweight processing"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = True,
                 g: int = 1,
                 e: float = 0.5,
                 k: int = 3,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c1, c_, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv3 = ConvModule(2 * c_, c2, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.m = nn.ModuleList(RepVGGDW(c_) for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through C3k layer."""
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        for m in self.m:
            y1 = m(y1)
        return self.cv3(torch.cat((y1, y2), 1))


class RepVGGDW(BaseModule):
    """RepVGG-style depthwise convolution"""

    def __init__(self, ed: int, k: int = 3, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(ed, ed, k, 1, k // 2, groups=ed, bias=False,
                               norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.conv1 = ConvModule(ed, ed, 1, 1, 0, groups=ed, bias=False,
                                norm_cfg=dict(type='BN'), act_cfg=None)
        self.dim = ed
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with RepVGG structure."""
        return torch.relu(self.conv(x) + self.conv1(x))


class Bottleneck(BaseModule):
    """Standard bottleneck block"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 shortcut: bool = True,
                 g: int = 1,
                 k: Tuple = (3, 3),
                 e: float = 0.5,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, k[0], 1, k[0] // 2, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through bottleneck."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(BaseModule):
    """Spatial Pyramid Pooling - Fast (SPPF) layer"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 k: int = 5,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_ * 4, c2, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SPPF layer."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


@MODELS.register_module(name='YOLOv11BackboneEnhance')
class YOLOv11BackboneEnhance(BaseModule):
    """YOLOv11 backbone optimized for small object detection

    Args:
        arch (str): Architecture type. Options are 'n', 's', 'm', 'l', 'x'.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
        norm_eval (bool): Whether to set norm layers to eval mode.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    # YOLOv11 architectures
    arch_settings = {
        'n': [
            [64, 128, 3, True, 0.25],  # C2fPSA
            [128, 256, 6, True, 0.25],  # C2fPSA
            [256, 512, 6, True, 0.5],  # C2fPSA
            [512, 1024, 3, True, 0.5],  # C2fPSA
        ],
        's': [
            [64, 128, 3, True, 0.33],
            [128, 256, 6, True, 0.33],
            [256, 512, 6, True, 0.5],
            [512, 1024, 3, True, 0.5],
        ],
        'm': [
            [96, 192, 3, True, 0.67],
            [192, 384, 6, True, 0.67],
            [384, 768, 6, True, 0.75],
            [768, 1536, 3, True, 0.75],
        ],
        'l': [
            [128, 256, 3, True, 0.5],
            [256, 512, 6, True, 0.5],
            [512, 1024, 6, True, 0.5],
            [1024, 2048, 3, True, 0.5],
        ],
        'x': [
            [160, 320, 3, True, 0.5],
            [320, 640, 6, True, 0.5],
            [640, 1280, 6, True, 0.5],
            [1280, 2560, 3, True, 0.5],
        ],
    }

    def __init__(self,
                 arch: str = 's',
                 out_indices: Sequence[int] = (2, 3, 4, 5),  # Include stride=2,4 for small objects
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        self.arch = arch
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        arch_setting = self.arch_settings[arch]
        base_channels = arch_setting[0][0]

        # Stem layer - modified for better small object detection
        self.stem = ConvModule(
            3, base_channels // 2, 3, 2, 1, bias=False,
            norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # High resolution layer for small objects (stride=2)
        self.hr_layer = ConvModule(
            base_channels // 2, base_channels // 2, 3, 1, 1, bias=False,
            norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # Additional high res layer (stride=2 -> stride=4)
        self.hr_conv = ConvModule(
            base_channels // 2, base_channels, 3, 2, 1, bias=False,
            norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # Stage 0: stride=4 -> stride=8
        self.stage0 = ConvModule(
            base_channels, arch_setting[0][0], 3, 2, 1, bias=False,
            norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # Build stages
        layers = []
        in_channels = arch_setting[0][0]

        for i, (mid_channels, out_channels, num_blocks, shortcut, expand_ratio) in enumerate(arch_setting):
            stage = []
            stage.append(ConvModule(
                in_channels, mid_channels, 3, 2, 1, bias=False,
                norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')))
            stage.append(C2fPSA(
                mid_channels, out_channels, num_blocks, shortcut,
                1, expand_ratio))

            layers.append(nn.Sequential(*stage))
            in_channels = out_channels

        self.stages = nn.ModuleList(layers)

        # SPPF layer
        self.sppf = SPPF(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function."""
        outs = []

        # Stem
        x = self.stem(x)  # stride=2

        if 0 in self.out_indices:
            outs.append(x)

        # High resolution layer for small objects
        x_hr = self.hr_layer(x)  # stride=2
        if 1 in self.out_indices:
            outs.append(x_hr)

        # stride=4 layer
        x = self.hr_conv(x_hr)  # stride=4
        if 2 in self.out_indices:
            outs.append(x)

        # stride=8
        x = self.stage0(x)  # stride=8
        if 3 in self.out_indices:
            outs.append(x)

        # Other stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if (i + 4) in self.out_indices:
                if i == len(self.stages) - 1:  # Last stage, add SPPF
                    x = self.sppf(x)
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i == 1:
                m = self.hr_layer
            elif i == 2:
                m = self.hr_conv
            elif i == 3:
                m = self.stage0
            else:
                m = self.stages[i - 4]

            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()