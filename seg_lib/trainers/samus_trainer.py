import os
import time

import torch
from seg_lib.eval.plots import plot_dataloader_sample
from seg_lib.trainers.seg_trainer import SegmentationModelTrainer

class SamusModelTrainer(SegmentationModelTrainer):
    def __format_prompt(self, batch, prompt_type):
        if prompt_type in batch and batch[prompt_type] is not None:
            return batch[prompt_type].to(device=self.data_config['device'])
        
        return None

    def train_step(self, batch):
        imgs = batch['image'].to(**self.data_config)
        low_masks = batch['low_mask'].to(**self.data_config)
        
        # forward
        preds = self.model({
            'image': imgs,
            'point_coords': self.__format_prompt(batch, 'point_coords'),
            'point_labels': self.__format_prompt(batch, 'point_labels'),
            'boxes': self.__format_prompt(batch, 'boxes'),
            'mask_inputs': None,
            'original_size': imgs.shape[-2:]
        }, multimask_output=False)
        train_loss = self.loss_f(preds['low_res_logits'], low_masks) 
        # backward
        self.optm.zero_grad()
        train_loss.backward()
        self.optm.step()

        return train_loss.item()
    
    @torch.no_grad()
    def val_step(self, batch):
        imgs = batch['image'].to(**self.data_config)
        low_masks = batch['low_mask'].to(**self.data_config)
        label = batch['label'].to(**self.data_config)
 
        start_time = time.time()
        preds = self.model({
            'image': imgs,
            'point_coords': self.__format_prompt(batch, 'point_coords'),
            'point_labels': self.__format_prompt(batch, 'point_labels'),
            'boxes': self.__format_prompt(batch, 'boxes'),
            'mask_inputs': None,
            'original_size': imgs.shape[-2:]
        }, multimask_output=False)
        end_time = time.time() - start_time
        val_loss = self.loss_f(preds['low_res_logits'], low_masks)

        self.update_metrics(label, preds['masks'])

        return val_loss.item(), end_time, low_masks.shape[0]
    
    def fit(self, *args, **kwargs):
        out_path = kwargs.get('output_path')
        plot_dataloader_sample(
            next(iter(args[0])),
            output_path=os.path.join(out_path, 'train_ds_sample.png')
        )
        plot_dataloader_sample(
            next(iter(args[1])),
            output_path=os.path.join(out_path, 'val_ds_sample.png')
        )
        return super().fit(*args, **kwargs)