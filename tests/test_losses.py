import torch
import pytest

from seg_lib.losses.combined_losses import (
    DiceAndBCELoss,
    MaskBCELoss,
    MaskDiceAndBCELoss,
)
from seg_lib.losses.dice_loss import DiceLoss, MaskDiceLoss
from seg_lib.losses.focal_loss import FocalLoss
from seg_lib.losses.selector import select_loss_by_model


def test_dice_and_bce_squeezes_4d_target_and_accepts_3d():
    loss_f = DiceAndBCELoss(classes=2)
    logits = torch.zeros(1, 2, 4, 4)
    target_3d = torch.zeros(1, 4, 4, dtype=torch.long)
    target_4d = torch.zeros(1, 1, 4, 4, dtype=torch.long)

    loss_3d = loss_f(logits, target_3d)
    loss_4d = loss_f(logits, target_4d)

    assert loss_3d.ndim == 0
    assert torch.isfinite(loss_3d)
    assert torch.isfinite(loss_4d)


def test_mask_dice_and_bce_reshapes_5d_and_accepts_long_4d():
    loss_f = MaskDiceAndBCELoss(pos_weight=torch.ones(1))
    logits_4d = torch.zeros(1, 1, 4, 4)
    target_4d = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    logits_5d = torch.zeros(1, 2, 1, 4, 4)
    target_5d = torch.ones(1, 2, 1, 4, 4, dtype=torch.long)

    loss_4d = loss_f(logits_4d, target_4d)
    loss_5d = loss_f(logits_5d, target_5d)

    assert torch.isfinite(loss_4d)
    assert torch.isfinite(loss_5d)
    assert target_4d.dtype == torch.long


def test_mask_bce_accepts_long_mask():
    loss_f = MaskBCELoss(pos_weight=torch.ones(1))
    loss = loss_f(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4, dtype=torch.long))
    assert torch.isfinite(loss)


def test_dice_loss_softmax_weights_and_shape_mismatch():
    loss_f = DiceLoss(n_classes=2)
    logits = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)

    without_softmax = loss_f(logits, target, softmax=False)
    with_softmax = loss_f(logits, target, softmax=True)
    weighted = loss_f(logits, target, weight=[0.2, 0.8], softmax=True)

    assert torch.isfinite(without_softmax)
    assert torch.isfinite(with_softmax)
    assert torch.isfinite(weighted)
    assert not torch.equal(without_softmax, weighted)

    with pytest.raises(TypeError):
        loss_f(logits, torch.zeros(1, 3, 3, dtype=torch.long), softmax=True)


def test_mask_dice_sigmoid_and_shape_mismatch():
    loss_f = MaskDiceLoss()
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)

    plain = loss_f(logits, target, sigmoid=False)
    activated = loss_f(logits, target, sigmoid=True)
    assert torch.isfinite(plain)
    assert torch.isfinite(activated)
    assert not torch.equal(plain, activated)

    with pytest.raises(TypeError):
        loss_f(logits, torch.ones(1, 1, 3, 3))


def test_focal_loss_alpha_branches_and_reduction():
    with pytest.raises(TypeError):
        FocalLoss(alpha=[0.2, 0.3], num_classes=3)
    with pytest.raises(TypeError):
        FocalLoss(alpha=1.0, num_classes=3)

    scalar = FocalLoss(alpha=0.25, gamma=2, num_classes=3, size_average=True)
    per_class = FocalLoss(alpha=[0.2, 0.3, 0.5], num_classes=3, size_average=False)
    preds = torch.randn(1, 3, 2, 2)
    labels = torch.randint(0, 3, (1, 2, 2))

    mean_loss = scalar(preds, labels)
    sum_loss = per_class(preds, labels)
    assert mean_loss.ndim == 0
    assert sum_loss.ndim == 0
    assert torch.isfinite(mean_loss)
    assert torch.isfinite(sum_loss)


def test_select_loss_by_model_names_and_cpu_device():
    samed = select_loss_by_model('SAMed', device=None, classes=2)
    msa = select_loss_by_model('MSA', device=None)
    default = select_loss_by_model('SAM', device='cpu')

    assert isinstance(samed, DiceAndBCELoss)
    assert isinstance(msa, MaskBCELoss)
    assert isinstance(default, MaskDiceAndBCELoss)
    assert msa.ce.pos_weight.device.type == 'cpu'
    assert default.ce.pos_weight.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
def test_select_loss_moves_pos_weight_to_cuda():
    loss_f = select_loss_by_model('SAM', device='cuda')
    assert loss_f.ce.pos_weight.device.type == 'cuda'
