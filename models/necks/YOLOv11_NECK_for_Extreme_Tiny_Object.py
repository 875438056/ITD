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


class C2fPSA(BaseModule):
    """C2f with Position-aware Spatial Attention (PSA) - for Neck"""

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
    """C3k2 Block - YOLOv11 innovation for efficient feature extraction"""

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
        self.cv2 = ConvModule(c1, 2 * self.c, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv3 = ConvModule(4 * self.c, c2, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        if c3k:
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g, e=1.0) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                                   for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        a, b = self.cv1(x).chunk(2, 1), self.cv2(x).chunk(2, 1)
        c = a[1]
        for m in self.m:
            c = m(c)
        return self.cv3(torch.cat((a[0], c, b[0], b[1]), 1))


class C3k(BaseModule):
    """C3k block for C3k2"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c1, c_, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv3 = ConvModule(2 * c_, c2, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepBottleneck(BaseModule):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, k[0], 1, k[0] // 2, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck(BaseModule):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(1, 3), e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, k[0], 1, k[0] // 2, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(BaseModule):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for improved feature extraction"""

    def __init__(self, c1: int, c2: int, k: int = 5, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.cv2 = ConvModule(c_ * 4, c2, 1, 1, 0, bias=False,
                              norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SmallObjectEnhancement(BaseModule):
    """Small object enhancement module for better detection of tiny objects"""

    def __init__(self, c1: int, c2: int, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        # Multi-scale dilated convolutions for small object features
        self.branch1 = ConvModule(c1, c2 // 4, 3, 1, 1, dilation=1, bias=False,
                                  norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.branch2 = ConvModule(c1, c2 // 4, 3, 1, 2, dilation=2, bias=False,
                                  norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.branch3 = ConvModule(c1, c2 // 4, 3, 1, 3, dilation=3, bias=False,
                                  norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        self.branch4 = ConvModule(c1, c2 // 4, 1, 1, 0, bias=False,
                                  norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # Feature fusion
        self.fusion = ConvModule(c2, c2, 1, 1, 0, bias=False,
                                 norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        # Attention for small objects
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 16, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2 // 16, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = F.interpolate(self.branch4(F.adaptive_avg_pool2d(x, 1)),
                           size=x.shape[2:], mode='nearest')

        out = self.fusion(torch.cat([b1, b2, b3, b4], dim=1))
        att = self.attention(out)
        return out * att


@MODELS.register_module()
class YOLOv11PAFPN(BaseModule):
    """YOLOv11 Path Aggregation Feature Pyramid Network optimized for small objects.

    This neck supports 5 feature levels including high-resolution features
    for better small object detection, incorporating YOLOv11 innovations like
    C2fPSA, C3k2, and SPPF.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_csp_blocks (int): Number of bottlenecks in C2fPSA. Default: 3.
        use_depthwise (bool): Whether to use depthwise separable convolution.
        use_c3k2 (bool): Whether to use C3k2 blocks. Default: True.
        upsample_cfg (dict): Config dict for interpolate layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: List[int] = [64, 128, 256, 512, 1024],  # stride 2,4,8,16,32
                 out_channels: int = 256,
                 num_csp_blocks: int = 3,
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

        # SPPF layer for the deepest feature (stride=32)
        self.sppf = SPPF(out_channels, out_channels, k=5)

        # Top-down path (FPN)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1, 0, -1):
            # Reduce layer for higher level features
            self.reduce_layers.append(
                conv(out_channels, out_channels, 1, 1, 0,
                     norm_cfg=norm_cfg, act_cfg=act_cfg))

            # Top-down fusion block
            if self.use_c3k2 and i <= 2:  # Use C3k2 for high-res features
                self.top_down_blocks.append(
                    C3k2(out_channels * 2, out_channels,
                         num_csp_blocks, c3k=True, shortcut=False, e=0.5))
            else:
                self.top_down_blocks.append(
                    C2fPSA(out_channels * 2, out_channels,
                           num_csp_blocks, shortcut=False, e=0.5))

        # Bottom-up path (PAN)
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            # Downsample layer
            self.downsamples.append(
                conv(out_channels, out_channels, 3, 2, 1,
                     norm_cfg=norm_cfg, act_cfg=act_cfg))

            # Bottom-up fusion block
            if self.use_c3k2 and i >= 2:  # Use C3k2 for deeper features
                self.bottom_up_blocks.append(
                    C3k2(out_channels * 2, out_channels,
                         num_csp_blocks, c3k=True, shortcut=False, e=0.5))
            else:
                self.bottom_up_blocks.append(
                    C2fPSA(out_channels * 2, out_channels,
                           num_csp_blocks, shortcut=False, e=0.5))

        # Small object enhancement layers for high-resolution features
        self.small_obj_enhance = nn.ModuleList([
            SmallObjectEnhancement(out_channels, out_channels),  # For stride=2
            SmallObjectEnhancement(out_channels, out_channels),  # For stride=4
            SmallObjectEnhancement(out_channels, out_channels),  # For stride=8
        ])

        # Upsampling layers for generating high-resolution outputs
        self.upsample = UpSample()

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            inputs: Features from backbone at different scales
                   Expected: (stride2, stride4, stride8, stride16, stride32)

        Returns:
            Tuple of output features for detection heads
            Returns: (stride2, stride4, stride8, stride16, stride32)
        """
        assert len(inputs) == self.num_levels

        # Align all features to the same channel dimension
        aligned_feats = []
        for i, feat in enumerate(inputs):
            aligned_feats.append(self.align_layers[i](feat))

        # Apply SPPF to the deepest feature
        aligned_feats[-1] = self.sppf(aligned_feats[-1])

        # Top-down path (FPN)
        fpn_feats = [aligned_feats[-1]]  # Start with deepest feature

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
        pan_feats = [fpn_feats[0]]  # Start with highest resolution

        for i in range(1, self.num_levels):
            # Downsample higher resolution feature
            down_feat = self.downsamples[i - 1](pan_feats[-1])

            # Concatenate and fuse
            fused_feat = torch.cat([fpn_feats[i], down_feat], dim=1)
            fused_feat = self.bottom_up_blocks[i - 1](fused_feat)

            pan_feats.append(fused_feat)

        # Apply small object enhancement to high-resolution features
        for i in range(min(3, len(pan_feats))):
            pan_feats[i] = self.small_obj_enhance[i](pan_feats[i])

        return tuple(pan_feats)