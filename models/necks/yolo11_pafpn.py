# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig


class Conv(BaseModule):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 d=1,
                 act=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3k2(BaseModule):
    """CSP Bottleneck with 2 convolutions - 修复版本"""

    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 e=0.5,
                 init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # 修复：创建n个3x3卷积块，而不是固定的3、5、7
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv1(x)
        # 修复：依次通过所有卷积块
        for conv in self.m:
            y1 = conv(y1)
        return self.cv3(torch.cat((y1, self.cv2(x)), dim=1))


@MODELS.register_module()
class YOLOv11PAFPN(BaseModule):
    """
    修改版PAFPN for YOLOv11，支持P1-P4输出

    输入：P1(stride=2), P2(stride=4), P3(stride=8), P4(stride=16)
    输出：P1_out, P2_out, P3_out, P4_out
    """

    def __init__(self,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 in_channels: List[int] = [64, 128, 256, 512],  # P1, P2, P3, P4
                 out_channels: List[int] = [64, 128, 256, 512],  # P1_out, P2_out, P3_out, P4_out
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        assert len(in_channels) == 4, "需要4个输入通道数对应P1-P4"
        assert len(out_channels) == 4, "需要4个输出通道数对应P1-P4"

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Helper for scaling depth
        d = lambda x: max(round(x * deepen_factor), 1)
        # Helper for scaling width
        w = lambda x: max(round(x * widen_factor), 8)

        # 使用widen_factor调整中间通道数
        self.lateral_channels = [w(c) for c in out_channels]

        # ================= 侧向连接 (Lateral connections) =================
        # 将backbone特征转换为统一的通道数
        self.lateral_conv1 = Conv(in_channels[0], self.lateral_channels[0], 1)  # P1
        self.lateral_conv2 = Conv(in_channels[1], self.lateral_channels[1], 1)  # P2
        self.lateral_conv3 = Conv(in_channels[2], self.lateral_channels[2], 1)  # P3
        self.lateral_conv4 = Conv(in_channels[3], self.lateral_channels[3], 1)  # P4

        # ================= Top-Down Path (FPN) =================
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # P4 -> P3 fusion
        self.fpn_conv3 = C3k2(
            self.lateral_channels[3] + self.lateral_channels[2],
            self.lateral_channels[2],
            n=d(2)
        )

        # P3 -> P2 fusion
        self.fpn_conv2 = C3k2(
            self.lateral_channels[2] + self.lateral_channels[1],
            self.lateral_channels[1],
            n=d(2)
        )

        # P2 -> P1 fusion
        self.fpn_conv1 = C3k2(
            self.lateral_channels[1] + self.lateral_channels[0],
            self.lateral_channels[0],
            n=d(2)
        )

        # ================= Bottom-Up Path (PAN) =================
        # P1 -> P2 fusion
        self.pan_downsample1 = Conv(self.lateral_channels[0], self.lateral_channels[0], 3, 2)
        self.pan_conv2 = C3k2(
            self.lateral_channels[0] + self.lateral_channels[1],
            self.lateral_channels[1],
            n=d(2)
        )

        # P2 -> P3 fusion
        self.pan_downsample2 = Conv(self.lateral_channels[1], self.lateral_channels[1], 3, 2)
        self.pan_conv3 = C3k2(
            self.lateral_channels[1] + self.lateral_channels[2],
            self.lateral_channels[2],
            n=d(2)
        )

        # P3 -> P4 fusion
        self.pan_downsample3 = Conv(self.lateral_channels[2], self.lateral_channels[2], 3, 2)
        self.pan_conv4 = C3k2(
            self.lateral_channels[2] + self.lateral_channels[3],
            self.lateral_channels[3],
            n=d(2)
        )

        # ================= 输出卷积 =================
        self.out_conv1 = Conv(self.lateral_channels[0], out_channels[0], 1)  # P1_out
        self.out_conv2 = Conv(self.lateral_channels[1], out_channels[1], 1)  # P2_out
        self.out_conv3 = Conv(self.lateral_channels[2], out_channels[2], 1)  # P3_out
        self.out_conv4 = Conv(self.lateral_channels[3], out_channels[3], 1)  # P4_out

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            inputs (Tuple[Tensor]): backbone输出的特征图
                期望顺序: (P1, P2, P3, P4) - 从高分辨率到低分辨率

        Returns:
            Tuple[Tensor]: PAFPN输出的特征图
                输出顺序: (P1_out, P2_out, P3_out, P4_out)
        """
        assert len(inputs) == 4, f"期望4个输入特征图，得到{len(inputs)}个"

        # 输入特征图：P1, P2, P3, P4
        p1, p2, p3, p4 = inputs

        # ================= 侧向连接 =================
        lat1 = self.lateral_conv1(p1)  # P1 lateral
        lat2 = self.lateral_conv2(p2)  # P2 lateral
        lat3 = self.lateral_conv3(p3)  # P3 lateral
        lat4 = self.lateral_conv4(p4)  # P4 lateral

        # ================= Top-Down Path (FPN) =================
        # P4 -> P3
        fpn4_up = self.upsample(lat4)
        fpn3 = self.fpn_conv3(torch.cat([fpn4_up, lat3], dim=1))

        # P3 -> P2
        fpn3_up = self.upsample(fpn3)
        fpn2 = self.fpn_conv2(torch.cat([fpn3_up, lat2], dim=1))

        # P2 -> P1
        fpn2_up = self.upsample(fpn2)
        fpn1 = self.fpn_conv1(torch.cat([fpn2_up, lat1], dim=1))

        # ================= Bottom-Up Path (PAN) =================
        # P1 -> P2
        pan1_down = self.pan_downsample1(fpn1)
        pan2 = self.pan_conv2(torch.cat([pan1_down, fpn2], dim=1))

        # P2 -> P3
        pan2_down = self.pan_downsample2(pan2)
        pan3 = self.pan_conv3(torch.cat([pan2_down, fpn3], dim=1))

        # P3 -> P4
        pan3_down = self.pan_downsample3(pan3)
        pan4 = self.pan_conv4(torch.cat([pan3_down, lat4], dim=1))

        # ================= 输出层 =================
        out1 = self.out_conv1(fpn1)  # P1_out (stride=2)
        out2 = self.out_conv2(pan2)  # P2_out (stride=4)
        out3 = self.out_conv3(pan3)  # P3_out (stride=8)
        out4 = self.out_conv4(pan4)  # P4_out (stride=16)

        return (out1, out2, out3, out4)
        # return (out2,)

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