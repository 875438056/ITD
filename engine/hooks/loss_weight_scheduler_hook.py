# F:/ITD/engine/hooks/loss_weight_scheduler_hook.py

from mmengine.hooks import Hook
from mmengine.runner import Runner
# 【【【第1步：导入HOOKS注册器】】】
from mmdet.registry import HOOKS


# 【【【第2步：在类的定义上方加上注册装饰器】】】
@HOOKS.register_module()
class LossWeightSchedulerHook(Hook):
    """
    一个用于在训练过程中动态调整特定损失权重的Hook。

    这个Hook会在每个epoch结束后，根据预设的调度计划（如线性增长），
    更新模型中某个损失函数（如 loss_centroid）的 loss_weight。

    Args:
        loss_name (str): 需要调整权重的损失项的名称，
                         例如 'loss_centroid' 或 'loss_ltrb'。
        start_epoch (int): 开始调整权重的epoch。
        end_epoch (int): 达到目标权重的epoch。
        start_weight (float): 初始的损失权重。
        end_weight (float): 最终的损失权重。
        schedule_type (str): 调度类型，目前支持 'linear'。

    usage:
         # 在你的模型配置文件中 (e.g., F:/ITD/configs/my_centernet_config.py)
         # ... (其他配置如 model, dataset, optimizer 等) ...
         # custom_imports = dict(imports=['ITD.engine.hooks.loss_weight_scheduler_hook'],
                                                            allow_failed_imports=False)
         # 添加自定义Hook配置
            custom_hooks = [
                dict(
                    type='LossWeightSchedulerHook',
                    loss_name='loss_centroid',  # 对应 LTRBCentroidNetDSNTHead 中的 loss_centroid
                    start_epoch=50,             # 从第50个epoch结束后开始调整
                    end_epoch=100,              # 到第100个epoch结束时达到最终权重
                    start_weight=0.05,          # 初始权重
                    end_weight=0.5              # 最终权重
                )
            ]

            # 如果原来还有其他 custom_hooks，将上面的dict加入到列表中即可
            # custom_hooks = [
            #     dict(type='原有的Hook1'),
            #     dict(type='LossWeightSchedulerHook', ...),
            # ]
    """

    def __init__(self,
                 loss_name: str,
                 start_epoch: int,
                 end_epoch: int,
                 start_weight: float,
                 end_weight: float,
                 schedule_type: str = 'linear'):
        self.loss_name = loss_name
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.schedule_type = schedule_type

        assert self.start_epoch < self.end_epoch, \
            'end_epoch 必须大于 start_epoch'

    def _after_train_epoch(self, runner: Runner) -> None:
        """在每个训练epoch结束后被调用。"""
        current_epoch = runner.epoch + 1  # runner.epoch 从0开始，我们通常说的epoch从1开始

        # 检查当前epoch是否在我们的调度区间内
        if self.start_epoch <= current_epoch <= self.end_epoch:
            # 根据线性调度计算当前的权重
            if self.schedule_type == 'linear':
                progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
                current_weight = self.start_weight + progress * (self.end_weight - self.start_weight)
            else:
                # 未来可以扩展其他调度方式，如指数增长等
                raise NotImplementedError(f'不支持的调度类型: {self.schedule_type}')

            # --- 核心逻辑：找到并修改模型中的损失权重 ---
            # CenterNet的head通常在 runner.model.bbox_head
            model_head = runner.model.bbox_head

            # 要修改的损失函数是head的一个属性，例如 model_head.loss_centroid
            if hasattr(model_head, self.loss_name):
                loss_module = getattr(model_head, self.loss_name)

                # 修改其 loss_weight 属性
                loss_module.loss_weight = current_weight

                # 在日志中打印信息，方便我们确认权重已更新
                runner.logger.info(
                    f'Epoch {current_epoch}: Updated {self.loss_name} weight to {current_weight:.4f}'
                )
            else:
                runner.logger.warning(
                    f'在 model.bbox_head 中未找到名为 {self.loss_name} 的损失模块，权重更新失败。'
                )