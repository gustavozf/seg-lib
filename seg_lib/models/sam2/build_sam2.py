# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from https://github.com/facebookresearch/sam2 (Apache-2.0).
# Hydra configs are expressed here as plain constructors so image inference
# does not depend on the upstream package.

import logging

import torch

from seg_lib.models.sam2.modelling.backbones.hieradet import Hiera
from seg_lib.models.sam2.modelling.backbones.image_encoder import FpnNeck, ImageEncoder
from seg_lib.models.sam2.modelling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)
from seg_lib.models.sam2.modelling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)
from seg_lib.models.sam2.modelling.position_encoding import PositionEmbeddingSine
from seg_lib.models.sam2.modelling.sam.transformer import RoPEAttention
from seg_lib.models.sam2.modelling.sam2_base import SAM2Base

logger = logging.getLogger(__name__)

SAM2_MODEL_TYPES = ("hiera_t", "hiera_s", "hiera_b+", "hiera_l")

# Fields omitted by an official YAML are left unset so Hiera's defaults apply.
# base+ is the one size that relies on those defaults.
_HIERA = {
    "hiera_t": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 7, 2),
        "global_att_blocks": (5, 7, 9),
        "window_pos_embed_bkg_spatial_size": (7, 7),
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "hiera_s": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 11, 2),
        "global_att_blocks": (7, 10, 13),
        "window_pos_embed_bkg_spatial_size": (7, 7),
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "hiera_b+": {
        "embed_dim": 112,
        "num_heads": 2,
        "backbone_channel_list": [896, 448, 224, 112],
    },
    "hiera_l": {
        "embed_dim": 144,
        "num_heads": 2,
        "stages": (2, 6, 36, 4),
        "global_att_blocks": (23, 33, 43),
        "window_pos_embed_bkg_spatial_size": (7, 7),
        "window_spec": (8, 4, 16, 8),
        "backbone_channel_list": [1152, 576, 288, 144],
    },
}

# Present on every SAM 2.1 config and absent from SAM 2.
_SAM21_FLAGS = {
    "no_obj_embed_spatial": True,
    "add_tpos_enc_to_obj_ptrs": True,
    "proj_tpos_enc_in_obj_ptrs": True,
    "use_signed_tpos_enc_to_obj_ptrs": True,
}

# Injected by upstream build_sam2 when apply_postprocessing=True (the default).
_DECODER_POSTPROCESS = {
    "dynamic_multimask_via_stability": True,
    "dynamic_multimask_stability_delta": 0.05,
    "dynamic_multimask_stability_thresh": 0.98,
}


def _rope(kv_in_dim=None, rope_k_repeat=False):
    kwargs = dict(
        rope_theta=10000.0,
        feat_sizes=(64, 64),
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
    )
    if rope_k_repeat:
        kwargs["rope_k_repeat"] = True
    if kv_in_dim is not None:
        kwargs["kv_in_dim"] = kv_in_dim
    return RoPEAttention(**kwargs)


def _memory_attention():
    layer = MemoryAttentionLayer(
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=_rope(),
        cross_attention=_rope(kv_in_dim=64, rope_k_repeat=True),
    )
    return MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=4,
    )


def _memory_encoder():
    return MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64,
            normalize=True,
            temperature=10000,
        ),
        mask_downsampler=MaskDownSampler(
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        fuser=Fuser(
            layer=CXBlock(
                dim=256,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True,
            ),
            num_layers=2,
        ),
    )


def _image_encoder(model_type: str):
    spec = _HIERA[model_type]
    trunk_kwargs = {
        key: spec[key]
        for key in (
            "embed_dim",
            "num_heads",
            "stages",
            "global_att_blocks",
            "window_pos_embed_bkg_spatial_size",
            "window_spec",
        )
        if key in spec
    }
    return ImageEncoder(
        scalp=1,
        trunk=Hiera(**trunk_kwargs),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                temperature=10000,
            ),
            d_model=256,
            backbone_channel_list=spec["backbone_channel_list"],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        ),
    )


def _build_sam2_model(model_type: str, sam21: bool) -> SAM2Base:
    if model_type not in _HIERA:
        raise ValueError(
            f'"{model_type}" should be one of: {SAM2_MODEL_TYPES}'
        )
    flags = dict(_SAM21_FLAGS) if sam21 else {"add_tpos_enc_to_obj_ptrs": False}
    return SAM2Base(
        image_encoder=_image_encoder(model_type),
        memory_attention=_memory_attention(),
        memory_encoder=_memory_encoder(),
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False,
        sam_mask_decoder_extra_args=dict(_DECODER_POSTPROCESS),
        **flags,
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys:
        logger.error(missing_keys)
        raise RuntimeError("SAM 2 checkpoint is missing keys.")
    if unexpected_keys:
        logger.error(unexpected_keys)
        raise RuntimeError("SAM 2 checkpoint has unexpected keys.")
    logger.info("Loaded SAM 2 checkpoint successfully")


def build_sam2(model_type: str = "hiera_b+", checkpoint=None, device: str = "cpu"):
    """Build a SAM 2 image model."""
    model = _build_sam2_model(model_type, sam21=False)
    _load_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()
    return model


def build_sam2_1(model_type: str = "hiera_b+", checkpoint=None, device: str = "cpu"):
    """Build a SAM 2.1 image model."""
    model = _build_sam2_model(model_type, sam21=True)
    _load_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()
    return model


sam2_model_registry = {
    "hiera_t": lambda checkpoint=None, device="cpu": build_sam2("hiera_t", checkpoint, device),
    "hiera_s": lambda checkpoint=None, device="cpu": build_sam2("hiera_s", checkpoint, device),
    "hiera_b+": lambda checkpoint=None, device="cpu": build_sam2("hiera_b+", checkpoint, device),
    "hiera_l": lambda checkpoint=None, device="cpu": build_sam2("hiera_l", checkpoint, device),
}

sam2_1_model_registry = {
    "hiera_t": lambda checkpoint=None, device="cpu": build_sam2_1("hiera_t", checkpoint, device),
    "hiera_s": lambda checkpoint=None, device="cpu": build_sam2_1("hiera_s", checkpoint, device),
    "hiera_b+": lambda checkpoint=None, device="cpu": build_sam2_1("hiera_b+", checkpoint, device),
    "hiera_l": lambda checkpoint=None, device="cpu": build_sam2_1("hiera_l", checkpoint, device),
}
