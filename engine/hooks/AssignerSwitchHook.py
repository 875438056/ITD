# Copyright (c) OpenMMLab. All rights reserved.
# This is a custom hook to switch the assigner during training.

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.logging import MMLogger

from mmdet.registry import TASK_UTILS


@HOOKS.register_module()
class AssignerSwitchHook(Hook):
    """
    A hook to switch the assigner of the bbox_head at a specific epoch.
    This is useful for implementing a "warm-up" assigner strategy.

    Args:
        switch_epoch (int): The epoch at which to switch the assigner.
        switch_assigner (dict): The configuration dictionary of the new
            assigner to switch to.
    """

    def __init__(self, switch_epoch: int, switch_assigner: dict):
        self.switch_epoch = switch_epoch
        self.switch_assigner_cfg = switch_assigner
        self._switched = False  # A flag to ensure the switch happens only once

    def before_train_epoch(self, runner) -> None:
        """
        Called before each training epoch.
        Checks if the current epoch is the switch epoch.
        """
        # Get current epoch. runner.epoch is 0-indexed.
        current_epoch = runner.epoch + 1

        # Check if it's time to switch and if we haven't switched yet
        if current_epoch != self.switch_epoch or self._switched:
            return

        # Get the logger to print info
        logger = MMLogger.get_current_instance()
        logger.info(f'--- AssignerSwitchHook: Epoch {current_epoch} reached. Switching assigner. ---')

        # Get the bbox_head from the model
        # Note: This assumes your model is a single-stage detector with a `bbox_head` attribute.
        # This is true for almost all detectors in MMDetection.
        if hasattr(runner.model, 'bbox_head'):
            bbox_head = runner.model.bbox_head
        else:
            # For models wrapped with DDP, etc.
            bbox_head = runner.model.module.bbox_head

        # Build the new assigner from the provided config
        new_assigner = TASK_UTILS.build(self.switch_assigner_cfg)

        # Get the old assigner's type for logging
        old_assigner_type = type(bbox_head.assigner).__name__

        # Replace the assigner in the bbox_head
        bbox_head.assigner = new_assigner

        # Log the successful switch
        logger.info(f"--> Successfully switched assigner from '{old_assigner_type}' "
                    f"to '{type(new_assigner).__name__}'.")

        # Set the flag to prevent switching again
        self._switched = True