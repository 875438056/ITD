import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from typing import List, Tuple


def autopad(k, p=None, d=1):
    """自动填充以获得相同的输出尺寸"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(BaseModule):
    """标准卷积层：Conv2d + BatchNorm + SiLU"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, init_cfg=None):
        super().__init__(init_cfg)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(BaseModule):
    """标准瓶颈层"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(BaseModule):
    """CSP瓶颈层，包含3个卷积层"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(BaseModule):
    """空间金字塔池化 - 快速版本"""

    def __init__(self, c1, c2, k=5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


@MODELS.register_module()
class YOLOv5sBackbone(BaseModule):
    """YOLOv5s主干网络实现

    Args:
        depth_multiple (float): 深度缩放因子，默认0.33
        width_multiple (float): 宽度缩放因子，默认0.50
        out_indices (Tuple[int]): 输出特征图的索引，默认(2, 4, 6, 9)
        frozen_stages (int): 冻结的阶段数，默认-1
        norm_eval (bool): 是否在eval模式下freeze BN，默认False
        init_cfg (dict): 初始化配置
    """

    # YOLOv5s架构配置 [from, number, module, args]
    arch_config = [
        [-1, 1, 'Conv', [64, 6, 2, 2]],  # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
        [-1, 3, 'C3', [128]],  # 2
        [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
        [-1, 6, 'C3', [256]],  # 4
        [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
        [-1, 9, 'C3', [512]],  # 6
        [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
        [-1, 3, 'C3', [1024]],  # 8
        [-1, 1, 'SPPF', [1024, 5]],  # 9
    ]

    def __init__(self,
                 depth_multiple=0.33,
                 width_multiple=0.50,
                 out_indices=(2, 4, 6, 9),
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # 构建网络层
        self.layers = nn.ModuleList()
        self._build_layers()

        # 冻结指定阶段
        self._freeze_stages()

    def _make_divisible(self, x, divisor=8):
        """使通道数能被divisor整除"""
        return max(divisor, int(x + divisor / 2) // divisor * divisor)

    def _build_layers(self):
        """构建网络层"""
        ch = [3]  # 输入通道数

        for i, (f, n, m, args) in enumerate(self.arch_config):
            # 获取模块类型
            if m == 'Conv':
                c1, c2 = ch[f], args[0]
                if c2 != 77:  # 不是最后一层
                    c2 = self._make_divisible(c2 * self.width_multiple)
                args = [c1, c2, *args[1:]]
                layer = Conv(*args)

            elif m == 'C3':
                c1, c2 = ch[f], args[0]
                c2 = self._make_divisible(c2 * self.width_multiple)
                args = [c1, c2, *args[1:]]
                if len(args) == 2:
                    args.append(max(round(n * self.depth_multiple), 1))
                else:
                    args[2] = max(round(args[2] * self.depth_multiple), 1)
                layer = C3(*args)

            elif m == 'SPPF':
                c1, c2 = ch[f], args[0]
                c2 = self._make_divisible(c2 * self.width_multiple)
                args = [c1, c2, *args[1:]]
                layer = SPPF(*args)

            # 添加层到模块列表
            self.layers.append(layer)
            ch.append(args[1])  # 输出通道数

    def _freeze_stages(self):
        """冻结指定阶段的参数"""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                if i < len(self.layers):
                    for param in self.layers[i].parameters():
                        param.requires_grad = False

    def forward(self, x):
        """前向传播"""
        outs = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        """重写train方法以支持norm_eval"""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 使用示例和配置
# if __name__ == "__main__":
#     # 创建模型
#     model = YOLOv5sBackbone()
#
#     # 测试前向传播
#     x = torch.randn(1, 3, 640, 640)
#     outputs = model(x)
#
#     print("YOLOv5s Backbone 输出特征图尺寸:")
#     for i, out in enumerate(outputs):
#         print(f"输出 {i}: {out.shape}")
#
#     # 计算模型参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     print(f"\n模型参数统计:")
#     print(f"总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
#
# # MMDetection配置文件示例
# backbone_config = {
#     'type': 'YOLOv5sBackbone',
#     'depth_multiple': 0.33,
#     'width_multiple': 0.50,
#     'out_indices': (2, 4, 6, 9),
#     'frozen_stages': -1,
#     'norm_eval': False,
#     'init_cfg': dict(
#         type='Kaiming',
#         layer='Conv2d',
#         a=0,
#         distribution='uniform',
#         mode='fan_in',
#         nonlinearity='leaky_relu'
#     )
# }