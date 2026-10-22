import matplotlib

matplotlib.use('Agg')

import numpy as np
import torch

from seg_lib.trainers.seg_trainer import SegmentationModelTrainer


class FakePredictor:
    """Stand-in for a promptable predictor. Records calls and returns fixed masks."""

    def __init__(self, height=4, width=4):
        self.mask_threshold = 0.0
        self.original_size = (height, width)
        self.device = torch.device('cpu')
        self.calls = []
        self.transform = self
        self.model = self

    def set_image(self, image):
        self.calls.append(('set_image', None if image is None else np.shape(image)))

    def reset_image(self):
        self.calls.append('reset')

    def apply_boxes_torch(self, boxes, original_size):
        self.calls.append(('boxes', tuple(boxes.shape), original_size))
        return boxes

    def apply_coords_torch(self, coords, original_size):
        self.calls.append(('coords', tuple(coords.shape), original_size))
        return coords

    def predict_torch(self, **kwargs):
        self.calls.append(('predict_torch', kwargs.get('boxes') is not None))
        height, width = self.original_size
        masks = torch.zeros(1, 3, height, width)
        masks[:, 1] = 1
        scores = torch.tensor([[0.1, 0.9, 0.2]])
        return masks, scores, None

    def predict(self, **kwargs):
        self.calls.append(('predict', kwargs))
        height, width = self.original_size
        masks = np.zeros((3, height, width), dtype=np.float32)
        masks[1] = 1
        scores = np.array([0.1, 0.9, 0.2], dtype=np.float32)
        return masks, scores, None


class ScriptedTrainer(SegmentationModelTrainer):
    """Trainer whose steps return scripted losses and Dice values."""

    def __init__(self, *args, batch_losses=None, dice_values=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_losses = list(batch_losses or [0.2, 0.6])
        self.dice_values = list(dice_values or [1.0])
        self.train_calls = 0
        self.val_calls = 0

    def train_step(self, batch):
        self._set_warmup_lr()
        self.train_calls += 1
        index = min(self.train_calls - 1, len(self.batch_losses) - 1)
        return self.batch_losses[index]

    def val_step(self, batch):
        self.val_calls += 1
        dice = self.dice_values.pop(0) if self.dice_values else 0.0
        self.metrics.ious.append(dice)
        self.metrics.dices.append(dice)
        self.metrics.maes.append(0.1)
        self.metrics.wfms.append(dice)
        self.metrics.emes.append(dice)
        return 0.4, 1.0, 1


def tiny_optimizer(lr=0.2):
    model = torch.nn.Linear(1, 1)
    return model, torch.optim.SGD(model.parameters(), lr=lr)
