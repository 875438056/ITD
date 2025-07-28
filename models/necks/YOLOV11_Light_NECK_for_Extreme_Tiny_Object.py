# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig


class UpSample(BaseModule):
    """Upsample layer for FPN."""

    def __init__(self, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return F.interpolate(x, scale_factor=2, mode='nearest')


class Concat(BaseModule):
    """Concatenate tensors along specified dimension."""

    def __init__(self, dimension: int = 1, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.d = dimension

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward function."""
        return torch.cat(x, self.d)


class LightC2f(BaseModule):
    """Lightweight C2f - reduced channels and simplified structure"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = False,
                 g: int = 1,
                 e: float = 0.25,  # Reduced from 0.5 to 0.25
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, 2 * self.c, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(2 * self.c, c2, 1, 1, 0, bias=False,  # Reduced concat channels
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        # Use lightweight bottlenecks
        self.m = nn.ModuleList(LightBottleneck(self.c, self.c, shortcut, g, e=1.0)
                               for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through LightC2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        # Only keep the last output to reduce memory
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat([y[0], y[-1]], 1))  # Only concat first and last


class LightBottleneck(BaseModule):
    """Lightweight Bottleneck without attention"""

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
        # Handle tuple kernel sizes properly
        k1 = k[0] if isinstance(k[0], int) else k[0][0]
        k2 = k[1] if isinstance(k[1], int) else k[1][0]
        self.cv1 = ConvModule(c1, c_, k1, 1, k1 // 2, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_, c2, k2, 1, k2 // 2, groups=g, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass without attention."""
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class SimplePSA(BaseModule):
    """Simplified Position-aware Spatial Attention - much lighter"""

    def __init__(self, c: int, reduction: int = 8, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.c = c
        # Reduce computation by using smaller hidden dimensions
        hidden_c = max(c // reduction, 16)
        self.cv1 = nn.Conv2d(c, hidden_c, 1, bias=False)
        self.cv2 = nn.Conv2d(hidden_c, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with simplified attention."""
        # Use global average pooling to reduce spatial dimensions
        b, c, h, w = x.shape
        # Simple channel attention instead of full spatial attention
        att = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        att = self.cv2(F.relu(self.cv1(att)))
        att = self.sigmoid(att)
        return x * att


class LightC3k2(BaseModule):
    """Lightweight C3k2 Block - reduced complexity"""

    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 c3k: bool = False,
                 e: float = 0.25,  # Reduced expansion ratio
                 g: int = 1,
                 shortcut: bool = True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, self.c, 1, 1, 0, bias=False,  # Reduced channels
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c1, self.c, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv3 = ConvModule(2 * self.c, c2, 1, 1, 0, bias=False,  # Simplified output
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # Use lightweight bottlenecks
        self.m = nn.ModuleList(LightBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
                               for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        a = self.cv1(x)
        b = self.cv2(x)

        # Process through bottlenecks
        for m in self.m:
            a = m(a)

        return self.cv3(torch.cat((a, b), 1))


class LightSPPF(BaseModule):
    """Lightweight SPPF - reduced pooling scales"""

    def __init__(self, c1: int, c2: int, k: int = 5, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_ * 3, c2, 1, 1, 0, bias=False,  # Reduced from 4 to 3
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)  # Only two pooling operations instead of three
        return self.cv2(torch.cat((x, y1, y2), 1))


class LightSmallObjectEnhancement(BaseModule):
    """Lightweight small object enhancement - reduced branches"""

    def __init__(self, c1: int, c2: int, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        # Only two branches instead of four
        self.branch1 = ConvModule(c1, c2 // 2, 3, 1, 1, dilation=1, bias=False,
                                  norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.branch2 = ConvModule(c1, c2 // 2, 3, 1, 2, dilation=2, bias=False,
                                  norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # Simplified attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 8, 1, bias=False),  # Reduced bottleneck
            nn.SiLU(inplace=True),
            nn.Conv2d(c2 // 8, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        out = torch.cat([b1, b2], dim=1)
        att = self.attention(out)
        return out * att


@MODELS.register_module()
class YOLOv11LightPAFPN(BaseModule):
    """Optimized YOLOv11 PAFPN with reduced memory usage.

    Optimizations:
    1. Reduced expansion ratios (e=0.25 instead of 0.5)
    2. Simplified attention mechanisms
    3. Fewer bottleneck blocks
    4. Reduced small object enhancement complexity
    5. Memory-efficient feature fusion

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_csp_blocks (int): Number of bottlenecks. Default: 2 (reduced from 3).
        use_depthwise (bool): Whether to use depthwise separable convolution.
        use_c3k2 (bool): Whether to use C3k2 blocks. Default: True.
        upsample_cfg (dict): Config dict for interpolate layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: List[int] = [64, 128, 256, 512, 1024],
                 out_channels: int = 256,
                 num_csp_blocks: int = 2,  # Reduced from 3 to 2
                 use_depthwise: bool = False,
                 use_c3k2: bool = True,
                 upsample_cfg: dict = dict(scale_factor=2, mode='nearest'),
                 norm_cfg: dict = dict(type='BN'),
                 act_cfg: dict = dict(type='SiLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_csp_blocks = num_csp_blocks
        self.upsample_cfg = upsample_cfg
        self.num_levels = len(in_channels)
        self.use_c3k2 = use_c3k2

        conv = ConvModule

        # Feature alignment layers for different scales
        self.align_layers = nn.ModuleList([
            conv(in_channels[i], out_channels, 1, 1, 0,
                 norm_cfg=norm_cfg, act_cfg=act_cfg)
            for i in range(self.num_levels)
        ])

        # Lightweight SPPF layer
        self.sppf = LightSPPF(out_channels, out_channels, k=5)

        # Top-down path (FPN) - use lightweight blocks
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1, 0, -1):
            # Reduce layer for higher level features
            self.reduce_layers.append(
                conv(out_channels, out_channels, 1, 1, 0,
                     norm_cfg=norm_cfg, act_cfg=act_cfg))

            # Use lightweight fusion blocks
            if self.use_c3k2 and i <= 2:
                self.top_down_blocks.append(
                    LightC3k2(out_channels * 2, out_channels,
                              num_csp_blocks, c3k=True, shortcut=False, e=0.25))
            else:
                self.top_down_blocks.append(
                    LightC2f(out_channels * 2, out_channels,
                             num_csp_blocks, shortcut=False, e=0.25))

        # Bottom-up path (PAN) - use lightweight blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            # Downsample layer
            self.downsamples.append(
                conv(out_channels, out_channels, 3, 2, 1,
                     norm_cfg=norm_cfg, act_cfg=act_cfg))

            # Use lightweight fusion blocks
            if self.use_c3k2 and i >= 2:
                self.bottom_up_blocks.append(
                    LightC3k2(out_channels * 2, out_channels,
                              num_csp_blocks, c3k=True, shortcut=False, e=0.25))
            else:
                self.bottom_up_blocks.append(
                    LightC2f(out_channels * 2, out_channels,
                             num_csp_blocks, shortcut=False, e=0.25))

        # Lightweight small object enhancement (only for first 2 levels to save memory)
        self.small_obj_enhance = nn.ModuleList([
            LightSmallObjectEnhancement(out_channels, out_channels),  # For stride=2
            LightSmallObjectEnhancement(out_channels, out_channels),  # For stride=4
        ])

        # Simple attention for remaining levels
        self.simple_enhance = nn.ModuleList([
            SimplePSA(out_channels, reduction=8)  # For stride=8
        ])

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Forward function with memory optimization.

        Args:
            inputs: Features from backbone at different scales

        Returns:
            Tuple of output features for detection heads
        """
        assert len(inputs) == self.num_levels

        # Align all features to the same channel dimension
        aligned_feats = []
        for i, feat in enumerate(inputs):
            aligned_feats.append(self.align_layers[i](feat))

        # Apply lightweight SPPF to the deepest feature
        aligned_feats[-1] = self.sppf(aligned_feats[-1])

        # Top-down path (FPN)
        fpn_feats = [aligned_feats[-1]]

        for i in range(self.num_levels - 2, -1, -1):
            # Reduce channels of deeper feature
            top_feat = self.reduce_layers[self.num_levels - 2 - i](fpn_feats[0])

            # Upsample deeper feature
            top_feat_up = F.interpolate(top_feat, size=aligned_feats[i].shape[2:],
                                        mode=self.upsample_cfg['mode'])

            # Concatenate and fuse
            fused_feat = torch.cat([aligned_feats[i], top_feat_up], dim=1)
            fused_feat = self.top_down_blocks[self.num_levels - 2 - i](fused_feat)

            fpn_feats.insert(0, fused_feat)

        # Bottom-up path (PAN)
        pan_feats = [fpn_feats[0]]

        for i in range(1, self.num_levels):
            # Downsample higher resolution feature
            down_feat = self.downsamples[i - 1](pan_feats[-1])

            # Concatenate and fuse
            fused_feat = torch.cat([fpn_feats[i], down_feat], dim=1)
            fused_feat = self.bottom_up_blocks[i - 1](fused_feat)

            pan_feats.append(fused_feat)

        # Apply lightweight enhancement selectively
        # Only enhance first 2 levels with complex enhancement
        for i in range(min(2, len(pan_feats))):
            pan_feats[i] = self.small_obj_enhance[i](pan_feats[i])

        # Use simple attention for stride=8
        if len(pan_feats) > 2:

            pan_feats[1] = self.simple_enhance[0](pan_feats[1])
            pan_feats[2] = self.simple_enhance[0](pan_feats[2])

        return tuple(pan_feats)