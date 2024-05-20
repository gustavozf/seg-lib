import gc
import os
import time

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from seg_lib.eval.metrics import Metrics

class SegmentationModelTrainer:
    DEFAULT_MASK_TH = 0.5

    def __init__(
        self,
        model: torch.nn.Module, optm: torch.optim.Optimizer,
        loss_f: torch.nn.Module, metrics: Metrics = Metrics(),
        device: str = 'cpu'
    ):
        self.model = model
        self.optm = optm
        self.loss_f = loss_f
        self.metrics = metrics
        self.iter_num = 0
        self.data_config = {'dtype': torch.float32, 'device': device}

    def train_step(self, batch):
        imgs = batch['image'].to(**self.data_config)
        masks = batch['label'].to(**self.data_config)

        # forward
        preds = self.model(imgs)
        train_loss = self.loss_f(preds, masks)

        # backward
        self.optm.zero_grad()
        train_loss.backward()
        self.optm.step()

        return train_loss.item()

    def train_loop(self, train_ds):
        self.model.train()
        num_train_batches = len(train_ds)

        train_losses = 0.0
        p_bar = tqdm(range(num_train_batches), desc='Train')
        for batch_idx, (batch) in enumerate(train_ds):
            train_loss = self.train_step(batch)
            train_losses += train_loss
            
            p_bar.update(1)
            batch_loss = train_loss
            mean_loss = train_losses / (batch_idx + 1)
            p_bar.set_postfix({'loss': batch_loss, 'mean_loss': mean_loss})
            self.iter_num += 1
        p_bar.close()

        # batch_idx + 1 == number of processed batches
        # train_losses / (batch_idx + 1) == mean train loss
        return train_losses / (batch_idx + 1)

    def update_metrics(self, masks, preds):
        masks = masks.detach().cpu().numpy()[:, 0, :, :]
        preds = preds.sigmoid().detach().cpu().numpy()[:, 0, :, :]
        preds = preds > self.DEFAULT_MASK_TH

        for j in range(preds.shape[0]):
            self.metrics.step(gt=masks[j], pred=preds[j])
    
    @torch.no_grad()
    def val_step(self, batch):
        imgs = batch['image'].to(**self.data_config)
        masks = batch['label'].to(**self.data_config)
        
        start_time = time.time()
        preds = self.model(imgs)
        end_time = time.time() - start_time

        val_loss = self.loss_f(preds, masks)
        self.update_metrics(masks, preds)

        return val_loss.item(), end_time, masks.shape[0]

    def val_loop(self, val_ds):
        self.model.eval()
        self.metrics.reset()
        num_val_batches = len(val_ds)

        sum_time = 0.0
        val_losses = 0.0
        n_processed_imgs = 0
        p_bar = tqdm(range(num_val_batches), desc='Validation')
        for batch in val_ds:
            val_loss, end_time, n_imgs = self.val_step(batch)
            
            val_losses += val_loss
            sum_time += end_time
            n_processed_imgs += n_imgs
            
            p_bar.update(1)
            p_bar.set_postfix({
                'val_loss': val_loss,
                'mean_val_loss': val_losses / n_processed_imgs,
                # mean dice considers the previous batches
                'mean_dice': self.metrics.get_results()['dice']
            })
        p_bar.close()

        return {
            'val_losses': val_losses / n_processed_imgs,
            'fps': n_processed_imgs / sum_time,
            'mean_latency': sum_time / num_val_batches,
            **self.metrics.get_results()
        }

    def fit(
            self,
            train_ds: DataLoader,
            val_ds: DataLoader,
            n_epochs: int = 200,
            eval_freq: int = 1,
            save_freq: int = 1,
            output_path: str = 'outputs/',
            model_name: str = 'seg_model'):
        # define the output paths
        best_model_path = os.path.join(output_path, f'best_{model_name}.pth')
        model_path = os.path.join(output_path, f'{model_name}.pth')
        logs_path = os.path.join(output_path, 'logs.csv')

        # create the dictionary to save the metrics
        log_zeros = torch.zeros(n_epochs + 1).numpy()
        logs = {
            'loss': log_zeros.copy(),
            'val_loss': log_zeros.copy(),
            'val_dice': log_zeros.copy(),
            'val_iou': log_zeros.copy(),
            'learning_rate': log_zeros.copy(),
            'has_improved': log_zeros.copy()
        }

        self.iter_num = 0
        best_dice = 0.0
        for epoch in range(n_epochs):
            print(f'Epoch {epoch+1}/{n_epochs}')
            logs['loss'][epoch] = self.train_loop(train_ds)
            logs['learning_rate'][epoch] = (
                self.optm.state_dict()['param_groups'][0]['lr']
            )

            #  evaluation
            if epoch % eval_freq == 0:
                metrics_dict = self.val_loop(val_ds)
                logs['val_loss'][epoch] = metrics_dict['val_losses']
                logs['val_dice'][epoch] = metrics_dict['dice']
                logs['val_iou'][epoch] = metrics_dict['iou']
                print(
                    f"Validation FPS: {metrics_dict['fps']} / "
                    f"Mean Latency: {metrics_dict['mean_latency']}"
                )
                
                if logs['val_dice'][epoch] > best_dice:
                    print(
                        'New best model found! '
                        f'{logs["val_dice"][epoch]} > {best_dice}'
                    )
                    best_dice = logs['val_dice'][epoch]
                    logs['has_improved'][epoch] = 1
                    torch.save(
                        self.model.state_dict(), best_model_path,
                        _use_new_zipfile_serialization=False)
                
                del metrics_dict

            if (epoch % save_freq == 0) or (epoch == n_epochs - 1):
                torch.save(
                    self.model.state_dict(), model_path,
                    _use_new_zipfile_serialization=False)
                pd.DataFrame(logs).to_csv(logs_path)
            gc.collect()
        
        pd.DataFrame(logs).to_csv(logs_path)
        return pd.DataFrame(logs)
