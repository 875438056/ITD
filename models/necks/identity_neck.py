# 文件路径: F:/ITD/models/necks/identity_neck.py (或 select_feature_neck.py)

from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class SelectFeatureNeck(BaseModule):
    """
    一个用于选择特定索引特征图的颈部模块。
    它从骨干网络输出的特征图列表中，根据 'in_index' 选择一个，
    并将其作为元组返回，以供后续的检测头使用。
    """
    def __init__(self, in_index=0):
        super().__init__()
        self.in_index = in_index

    def forward(self, inputs):
        # 确保输入是列表或元组
        assert isinstance(inputs, (list, tuple))
        # 根据索引选择特征图
        selected_feature = inputs[self.in_index]
        # CenterNetHead 期望接收一个元组(tuple)作为输入，所以我们将选择的特征图包裹成元组
        return (selected_feature,)