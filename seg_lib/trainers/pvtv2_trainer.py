import os
import time

import torch
from seg_lib.trainers.seg_trainer import SegmentationModelTrainer

class PVTv2ModelTrainer(SegmentationModelTrainer):
    def train_step(self, batch):
        imgs = batch['image'].to(**self.data_config)
        masks = batch['label'].to(**self.data_config)

        # forward
        preds = self.model(imgs)

        # for outputs with more than 1 output
        if isinstance(preds, tuple):
            train_loss = self.loss_f(preds[0], masks)
            for i in range(1, len(preds)):
                train_loss = train_loss + self.loss_f(preds[i], masks)
        else:
            train_loss = self.loss_f(preds, masks)

        # backward
        self.optm.zero_grad()
        train_loss.backward()
        self.optm.step()

        return train_loss.item()
    
    @torch.no_grad()
    def val_step(self, batch):
        imgs = batch['image'].to(**self.data_config)
        masks = batch['label'].to(**self.data_config)
        
        start_time = time.time()
        preds = self.model(imgs)
        end_time = time.time() - start_time

        # for outputs with more than 1 output
        if isinstance(preds, tuple):
            final_preds = preds[0]
            val_loss = self.loss_f(final_preds, masks)
            for i in range(1, len(preds)):
                final_preds = final_preds + preds[i]
                val_loss = val_loss + self.loss_f(preds[i], masks)

            preds = final_preds
        else:
            val_loss = self.loss_f(preds, masks)

        self.update_metrics(masks, preds)

        return val_loss.item(), end_time, masks.shape[0]