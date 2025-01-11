from typing import Optional
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
# from mmdet.registry import HOOKS
from mmengine.registry import HOOKS
    
@HOOKS.register_module()
class EpochsHook(Hook):
    def before_train_epoch(self, runner):
        from .unifyos import UnifiedObjectSample
        try:
            lens = len(runner.train_dataloader.dataset.dataset.pipeline.transforms)
            for i in range(lens):
                if isinstance(runner.train_dataloader.dataset.dataset.pipeline.transforms[i], UnifiedObjectSample):
                    runner.train_dataloader.dataset.dataset.pipeline.transforms[i].cur_epoch = runner.epoch + 1
                    print("cur_epoch in UnifiedObjectSample has been updated:{}!".format(runner.epoch + 1))
        except:
            import logging
            logging.error("EpochHook set ObjectSample's cur_epoch:{} failed!".format(runner.epoch + 1))