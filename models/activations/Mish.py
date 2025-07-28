# F:\ITD\models\activations\Mish.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class Mish(nn.Module):
    """Mish Activation Function.

    .. math::
        Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    def __init__(self, **kwargs):  # <--- 修改这里！使用 **kwargs
        super().__init__()
        # Mish 激活函数本身不支持原地操作(inplace)，
        # 但为了与 ConvModule 的接口兼容，我们接收所有传入的参数但忽略它们。
        # **kwargs 会捕获像 inplace=True 这样的参数，但我们不在函数体里使用它。

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))