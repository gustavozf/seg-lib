import os

import torch
import pytest

from tests.conftest import ScriptedTrainer, tiny_optimizer
from seg_lib.losses.combined_losses import MaskBCELoss
from seg_lib.trainers.pvtv2_trainer import PVTv2ModelTrainer
from seg_lib.trainers.samus_trainer import SamusModelTrainer
from seg_lib.trainers.seg_trainer import SegmentationModelTrainer


def test_warmup_period_base_lr_and_completed_ramp():
    model, opt = tiny_optimizer(lr=0.2)
    idle = SegmentationModelTrainer(model, opt, torch.nn.MSELoss(), warmup_period=0)
    idle._set_warmup_lr()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.2)

    missing = SegmentationModelTrainer(
        model, opt, torch.nn.MSELoss(), base_lr=None, warmup_period=4
    )
    missing._set_warmup_lr()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.2)

    trainer = SegmentationModelTrainer(
        model, opt, torch.nn.MSELoss(), base_lr=0.2, warmup_period=4
    )
    trainer.iter_num = 0
    trainer._set_warmup_lr()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.05)
    trainer.iter_num = 1
    trainer._set_warmup_lr()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.1)
    opt.param_groups[0]['lr'] = 0.2
    trainer.iter_num = 4
    trainer._set_warmup_lr()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.2)


def test_train_and_val_loops_use_batch_means():
    model, opt = tiny_optimizer()
    trainer = ScriptedTrainer(
        model, opt, torch.nn.MSELoss(), batch_losses=[0.2, 0.6]
    )
    assert trainer.train_loop([None, None]) == pytest.approx(0.4)

    result = trainer.val_loop([None, None])
    assert result['val_losses'] == pytest.approx(0.4)
    assert trainer.val_calls == 2

    empty = trainer.val_loop([])
    assert empty['val_losses'] == 0.0
    assert empty['fps'] == 0.0
    assert empty['mean_latency'] == 0.0
    assert 'dice' in empty


def test_fit_eval_frequency_best_checkpoint_and_unwrapped_save(tmp_path, monkeypatch):
    model, opt = tiny_optimizer()
    trainer = ScriptedTrainer(
        model,
        opt,
        torch.nn.MSELoss(),
        dice_values=[0.9, 0.2],
    )
    wrapped = torch.nn.DataParallel(model)
    trainer.model = wrapped
    saved = []

    def record_save(state, path, **kwargs):
        saved.append((path, list(state)))

    monkeypatch.setattr('seg_lib.trainers.seg_trainer.torch.save', record_save)
    logs = trainer.fit(
        [None, None],
        [None],
        n_epochs=4,
        eval_freq=2,
        save_freq=2,
        output_path=str(tmp_path),
        model_name='net',
    )
    assert trainer.val_calls == 2
    best_saves = [item for item in saved if item[0].endswith('best_net.pth')]
    assert len(best_saves) == 1
    assert logs['has_improved'].sum() == 1
    assert not any(key.startswith('module.') for key in best_saves[0][1])
    model_saves = [item for item in saved if os.path.basename(item[0]) == 'net.pth']
    # epochs 0 and 2 match save_freq; epoch 3 is the last epoch
    assert len(model_saves) == 3


class _PromptNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, batch, multimask_output=False):
        self.seen.append(batch['point_coords'])
        logits = batch['image'][:, :1] + self.bias
        return {'low_res_logits': logits, 'masks': torch.sigmoid(logits)}


def _samus_batch(with_prompt):
    batch = {
        'image': torch.zeros(1, 3, 4, 4),
        'low_mask': torch.zeros(1, 1, 4, 4, dtype=torch.long),
        'label': torch.zeros(1, 1, 4, 4, dtype=torch.long),
    }
    if with_prompt:
        batch['point_coords'] = torch.zeros(1, 1, 2)
        batch['point_labels'] = torch.ones(1, 1, dtype=torch.int)
        batch['boxes'] = torch.zeros(1, 4)
    return batch


def test_samus_train_step_with_and_without_prompts():
    model = _PromptNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    trainer = SamusModelTrainer(
        model,
        opt,
        MaskBCELoss(pos_weight=torch.ones(1)),
        base_lr=0.2,
        warmup_period=4,
    )
    trainer.train_step(_samus_batch(False))
    trainer.iter_num = 0
    trainer.train_step(_samus_batch(True))
    assert model.seen[0] is None
    assert model.seen[1] is not None
    assert opt.param_groups[0]['lr'] == pytest.approx(0.05)


def test_samus_fit_default_and_custom_output_path(tmp_path, monkeypatch):
    plotted = []
    monkeypatch.setattr(
        'seg_lib.trainers.samus_trainer.plot_dataloader_sample',
        lambda batch, output_path=None, **kwargs: plotted.append(output_path),
    )
    monkeypatch.chdir(tmp_path)
    os.makedirs('outputs', exist_ok=True)
    model = _PromptNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = SamusModelTrainer(
        model, opt, MaskBCELoss(pos_weight=torch.ones(1))
    )
    trainer.train_step = lambda batch: 0.1
    trainer.val_step = lambda batch: (0.1, 1.0, 1)
    batch = _samus_batch(False)
    trainer.fit([batch], [batch], n_epochs=1, eval_freq=1, save_freq=1)
    assert plotted[0].replace('\\', '/').endswith('outputs/train_ds_sample.png')
    assert plotted[1].replace('\\', '/').endswith('outputs/val_ds_sample.png')

    custom = tmp_path / 'custom'
    custom.mkdir()
    plotted.clear()
    trainer.fit(
        [batch],
        [batch],
        n_epochs=1,
        output_path=str(custom),
    )
    assert plotted[0].replace('\\', '/').endswith('custom/train_ds_sample.png')


class _Heads(torch.nn.Module):
    def __init__(self, twin):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.twin = twin

    def forward(self, imgs):
        out = imgs[:, :1] * self.scale
        if self.twin:
            return out, out + 0.0
        return out


class _CountingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pred, target):
        self.calls += 1
        return pred.float().mean()


def test_pvtv2_sums_heads_and_applies_warmup():
    image = torch.ones(1, 3, 4, 4)
    batch = {
        'image': image,
        'label': torch.zeros(1, 1, 4, 4, dtype=torch.long),
    }
    twin = _Heads(True)
    opt = torch.optim.SGD(twin.parameters(), lr=0.2)
    loss_f = _CountingLoss()
    trainer = PVTv2ModelTrainer(
        twin, opt, loss_f, base_lr=0.2, warmup_period=4
    )
    trainer.train_step(batch)
    assert loss_f.calls == 2
    assert opt.param_groups[0]['lr'] == pytest.approx(0.05)

    single = _Heads(False)
    single_loss = _CountingLoss()
    single_trainer = PVTv2ModelTrainer(
        single, torch.optim.SGD(single.parameters(), lr=0.2), single_loss
    )
    single_trainer.train_step(batch)
    assert single_loss.calls == 1

    fresh = _Heads(True)
    fresh_loss = _CountingLoss()
    fresh_trainer = PVTv2ModelTrainer(
        fresh, torch.optim.SGD(fresh.parameters(), lr=0.2), fresh_loss
    )
    val_loss, _, count = fresh_trainer.val_step(batch)
    assert fresh_loss.calls == 2
    assert count == 1
    assert val_loss == pytest.approx(2.0)
