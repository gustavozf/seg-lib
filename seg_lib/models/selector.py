from argparse import Namespace
from typing import List

import torch
from seg_lib.models.sam import sam_model_registry, SamPredictor
from seg_lib.models.samus import samus_model_registry, SamusPredictor

SUPPORTED_TOPOLOGIES = {'SAM', 'SAMUS'}
SUPPORTED_MODEL_TYPES = {'default', 'vit_l', 'vit_b'}

def build_sam_predictor(
        checkpoint_path: str,
        model_type: str = 'default',
        device: str = 'cpu'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)

def build_samus_predictor(checkpoint_path: List[str], device: str = 'cpu'):
    sam_path, samus_path = checkpoint_path
    samus = samus_model_registry['vit_b'](
        encoder_input_size=256, checkpoint=sam_path
    )
    samus.to(device)
    checkpoint = torch.load(samus_path, map_location=torch.device(device))
    samus.load_state_dict(checkpoint)
    return SamusPredictor(samus)

def predictor_selector(
        checkpoint_path: str,
        model_topology: str = 'SAM',
        model_type: str = 'default',
        device: str = 'cpu'
):
    if model_topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(
            f'"{model_topology}" should be one of: {SUPPORTED_TOPOLOGIES}'
        )
    
    if model_topology ==  'SAMUS':
        return build_samus_predictor(checkpoint_path, device=device)

    return build_sam_predictor(
        checkpoint_path, model_type=model_type, device=device
    )