# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# ---------------------------------------------------------------
# MaskTwins
# - Add function: round_score, find_indices_of_score, add_random
# - Modify EvalHook: save the best n checkpoint


import os
import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm

def round_score(score, decimals=4):
    return round(score, decimals)

def find_indices_of_score(scores, score):
    return [index for index, s in enumerate(scores) if round_score(s) == round_score(score)]

def add_random(score):
    import random
    return score+1e-7*random.randint(100, 499)



class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, save_n=10,**kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.best_scores = []
        self.best_ckpt_paths = {}
        self.save_n = save_n # max num of saved checkpoint
        self.flag = False  # the same score or not
        self.samedict = {}

    def _judge_full(self):
        if len(self.best_scores) < self.save_n:
            return False
        elif not self.flag:
            return True
        elif len(self.best_scores)<(self.save_n - len(self.samedict)+sum(self.samedict.values())):
            return False
        else:
            return True

    
    def _save_ckpt(self, runner, key_score):
        
        import os

        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
        else:
            current = f'iter_{runner.iter + 1}'

        if not self._judge_full():
            if key_score in self.best_scores:
                if not self.flag:
                    self.flag = True
                self.samedict[key_score] = self.samedict.get(key_score, 1) + 1
                key_score = add_random(key_score)
                while key_score in self.best_scores:
                    key_score = add_random(key_score)
            self.best_scores.append(key_score)
            self.best_ckpt_paths[key_score] = self._save_checkpoint(runner, current)
        else:
            min_score = min(self.best_scores)
            min_indices = find_indices_of_score(self.best_scores, min_score)

            if key_score >= min_score:
                
                if key_score > min_score:
                    for index in sorted(min_indices, reverse=True):
                        del_score = self.best_scores.pop(index)
                        del_path = self.best_ckpt_paths.pop(del_score)
                        if osp.isfile(del_path):
                            os.remove(del_path)

                    if len(min_indices) > 1:
                        del self.samedict[min_score]
                        if len(self.samedict) == 0:
                            self.flag = False

                if key_score in self.best_scores:
                    if not self.flag:
                        self.flag = True
                    self.samedict[key_score] = self.samedict.get(key_score, 1) + 1
                    key_score = add_random(key_score)
                    while key_score in self.best_scores:
                        key_score = add_random(key_score)
                self.best_scores.append(key_score)
                self.best_ckpt_paths[key_score] = self._save_checkpoint(runner, current)
    
    def _save_checkpoint(self, runner,current):
        ckpt_name = f'best_{self.key_indicator}_{current}.pth'
        ckpt_path = osp.join(runner.work_dir, ckpt_name)
        
        runner.save_checkpoint(
                runner.work_dir, ckpt_name, create_symlink=False)
        runner.logger.info(
                f'Now best checkpoint is saved as {ckpt_name}.')
        return ckpt_path

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """


    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        # print('--------------------distevalhook---------------------')
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
