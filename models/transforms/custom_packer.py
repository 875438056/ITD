# F:/ITD/models/transforms/custom_packer.py

from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS
from mmengine.structures import InstanceData


@TRANSFORMS.register_module()
class PackCustomDetInputs(PackDetInputs):
    """
    一个自定义的打包模块，它继承自 PackDetInputs，
    并额外打包 'gt_centroids' 数据。
    """

    def transform(self, results: dict) -> dict:
        """
        重写 transform 方法。
        首先调用父类的方法来完成所有标准数据的打包，
        然后在此基础上，手动添加我们的自定义数据。
        """
        # 1. 调用父类的方法，完成标准打包工作
        packed_results = super().transform(results)

        # 2. 【【【 核心修正 】】】
        # 在单张图片的处理阶段，'data_samples' 的值是单个 DetDataSample 对象，不是列表。
        # 因此我们直接获取它，而不是用索引 [0]。
        data_sample = packed_results['data_samples']

        # 3. 从输入 results 中获取我们自定义的 gt_centroids
        if 'gt_centroids' in results:
            # 确保 gt_instances 存在
            if 'gt_instances' not in data_sample:
                data_sample.gt_instances = InstanceData()

            # 设置自定义属性
            data_sample.gt_instances.gt_centroids = results['gt_centroids']

        return packed_results