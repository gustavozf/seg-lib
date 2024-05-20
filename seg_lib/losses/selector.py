import torch

from seg_lib.losses.combined_losses import (
    DiceAndBCELoss, MaskBCELoss, MaskDiceAndBCELoss
)

def select_loss_by_model(
        model_name: str='SAM', device: str=None, classes: int=2
    ):
    if model_name == "SAMed":
        return DiceAndBCELoss(classes=classes)
    
    pos_weight = torch.ones([1]) * 2
    if 'cuda' in device:
        pos_weight = pos_weight.cuda(device=torch.device(device))
    
    if model_name == "MSA":
        return MaskBCELoss(pos_weight=pos_weight)
    
    return MaskDiceAndBCELoss(pos_weight=pos_weight)