from argparse import Namespace

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import variant_to_config_mapping

from seg_lib.models.cafe_net.pvt import CAFE
from seg_lib.models.pvt_v2.pvt_v2_seg import SegPVT
from seg_lib.models.sam import sam_model_registry, SamPredictor
from seg_lib.models.sam_med2d import (
    sam_med_2d_model_registry, SamMed2DPredictor
)
from seg_lib.models.samus import samus_model_registry, SamusPredictor

SEG_MODELS = {
    'SegPVT': lambda c: SegPVT(backbone_ckpt_path=c),
    'CAFE': lambda c: CAFE(pvtv2_path=c)
}
SAM_PREDICTORS = {
    'SAM': SamPredictor,
    'SAMv2': SAM2ImagePredictor,
    'SAM-Med2D': SamMed2DPredictor,
    'SAMUS': SamusPredictor
}
SAM_INPUT_SIZES = {
    'SAM': {'input_size': 1024, 'embedding_size': 256},
    'SAM-Med2D': {'input_size': 256, 'embedding_size': 256},
    'SAMUS': {'input_size': 256, 'embedding_size': 128}
}

SUPPORTED_SEG_MODELS = set(SEG_MODELS.keys())
SUPPORTED_SAM_MODELS = set(SAM_PREDICTORS.keys())
SUPPORTED_MODEL_TYPES = {'default', 'vit_h' 'vit_l', 'vit_b'}
SUPPORTED_SAMv2_TYPES = set(variant_to_config_mapping.keys())

def seg_selector(
        checkpoint_path: str,
        model_topology: str = 'SegPVT',
        device: str = 'cpu'):
    if model_topology not in SUPPORTED_SEG_MODELS:
        raise ValueError(
            f'"{model_topology}" should be one of: {SUPPORTED_SEG_MODELS}'
        )
    model = SEG_MODELS[model_topology](checkpoint_path)
    model.to(device)
    return model

def build_sam_model(
        checkpoint_path: str,
        model_type: str = 'default',
        device: str = 'cpu'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam

def build_samv2_model(
        checkpoint_path: str,
        model_type: str = 'default',
        device: str = 'cpu'):
    samv2 = build_sam2(variant_to_config_mapping[model_type], checkpoint_path)
    samv2.to(device)
    return samv2

def build_sam_med_2d_model(checkpoint_path: str, device: str = 'cpu'):
    args = Namespace()
    args.image_size = SAM_INPUT_SIZES['SAM-Med2D']['input_size']
    args.encoder_adapter = True
    args.sam_checkpoint = checkpoint_path
    sam_med_2d = sam_med_2d_model_registry['vit_b'](args)
    sam_med_2d.to(device)
    return sam_med_2d

def build_samus_model(checkpoint_path: str, device: str = 'cpu'):
    samus = samus_model_registry['vit_b'](
        encoder_input_size=SAM_INPUT_SIZES['SAMUS']['input_size'],
        checkpoint=checkpoint_path
    )
    samus.to(device)
    return samus

def sam_selector(
    checkpoint_path: str,
    model_topology: str = 'SAM',
    model_type: str = 'default',
    device: str = 'cpu'):
    if model_topology not in SUPPORTED_SAM_MODELS:
        raise ValueError(
            f'"{model_topology}" should be one of: {SUPPORTED_SAM_MODELS}'
        )

    if model_topology ==  'SAMUS':
        return build_samus_model(checkpoint_path, device=device)
    if model_topology ==  'SAM-Med2D':
        return build_sam_med_2d_model(checkpoint_path, device=device)
    if model_topology ==  'SAMv2':
        return build_samv2_model(
            checkpoint_path, model_type=model_type, device=device
        )

    return build_sam_model(
        checkpoint_path, model_type=model_type, device=device
    )

def predictor_selector(
        checkpoint_path: str,
        model_topology: str = 'SAM',
        model_type: str = 'default',
        device: str = 'cpu'
):
    if model_topology not in SUPPORTED_SAM_MODELS:
        raise ValueError(
            f'"{model_topology}" should be one of: {SUPPORTED_SAM_MODELS}'
        )
    
    model = sam_selector(
        checkpoint_path,
        model_topology=model_topology,
        model_type=model_type,
        device=device)
    return SAM_PREDICTORS[model_topology](model)
