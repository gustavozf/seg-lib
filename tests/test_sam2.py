import numpy as np
import pytest
import torch
from PIL import Image

from seg_lib.models.sam2.build_sam2 import (
    _load_checkpoint,
    build_sam2,
    build_sam2_1,
    sam2_model_registry,
)
from seg_lib.models.sam2.predictor import Sam2Predictor


class _PromptEncoder:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, points, boxes, masks):
        self.owner.prompt_calls.append((points, boxes, masks))
        batch = 1 if points is None else points[0].shape[0]
        return torch.zeros(batch, 1, 4), torch.zeros(batch, 4, 4, 4)

    def get_dense_pe(self):
        return torch.zeros(1, 4, 4, 4)


class _FakeSam2:
    image_size = 32
    directly_add_no_mem_embed = True
    no_mem_embed = torch.zeros(1)

    def __init__(self):
        self.prompt_calls = []
        self.decoder_calls = []
        self.sam_prompt_encoder = _PromptEncoder(self)

    @property
    def device(self):
        return torch.device('cpu')

    def forward_image(self, image):
        return image

    def _prepare_backbone_features(self, backbone_out):
        batch = backbone_out.shape[0]
        feats = []
        for size in (256, 128, 64):
            feats.append(torch.zeros(size * size, batch, 4))
        return None, feats, None, None

    def sam_mask_decoder(self, **kwargs):
        self.decoder_calls.append(kwargs)
        batch = kwargs['sparse_prompt_embeddings'].shape[0]
        low_res = torch.zeros(batch, 1, 8, 8)
        scores = torch.zeros(batch, 1)
        return low_res, scores, None, None


def _predictor():
    model = _FakeSam2()
    return Sam2Predictor(model), model


def test_build_rejects_unknown_type_and_bad_checkpoints(tmp_path):
    with pytest.raises(ValueError):
        build_sam2('not-a-model')

    model = sam2_model_registry['hiera_t'](checkpoint=None, device='cpu')
    assert model.use_signed_tpos_enc_to_obj_ptrs is False
    assert model.add_tpos_enc_to_obj_ptrs is False

    missing = tmp_path / 'missing.pt'
    torch.save({'model': {}}, missing)
    with pytest.raises(RuntimeError, match='checkpoint is missing keys'):
        _load_checkpoint(model, str(missing))

    extra = model.state_dict()
    extra['not_a_parameter'] = torch.zeros(1)
    unexpected = tmp_path / 'extra.pt'
    torch.save({'model': extra}, unexpected)
    with pytest.raises(RuntimeError, match='checkpoint has unexpected keys'):
        _load_checkpoint(model, str(unexpected))

    valid = tmp_path / 'valid.pt'
    torch.save({'model': model.state_dict()}, valid)
    _load_checkpoint(model, str(valid))
    _load_checkpoint(model, None)

    sam21 = build_sam2_1('hiera_t', checkpoint=None, device='cpu')
    assert sam21.use_signed_tpos_enc_to_obj_ptrs is True
    assert sam21.add_tpos_enc_to_obj_ptrs is True


def test_predictor_image_setup_and_prompt_paths():
    predictor, model = _predictor()
    with pytest.raises(RuntimeError):
        predictor.predict(point_coords=np.array([[1.0, 1.0]]), point_labels=np.array([1]))
    with pytest.raises(RuntimeError):
        predictor.get_image_embedding()

    with pytest.raises(NotImplementedError):
        predictor.set_image('not-an-image')

    predictor._transforms = lambda image: torch.zeros(1, 8, 8)
    with pytest.raises(ValueError):
        predictor.set_image(np.zeros((8, 8, 3), dtype=np.uint8))

    predictor, model = _predictor()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    predictor.set_image(image)
    assert predictor._is_image_set
    embedding = predictor.get_image_embedding()
    assert embedding.shape[-2:] == (64, 64)

    predictor.set_image(Image.new('RGB', (8, 8)))
    assert predictor._orig_hw == [(8, 8)]

    def run(kind, **kwargs):
        model.prompt_calls.clear()
        masks, scores, low_res = predictor.predict(**kwargs)
        assert masks.ndim == 3
        assert scores.ndim == 1
        assert low_res.ndim == 3
        return model.prompt_calls[-1]

    points = run(
        'points',
        point_coords=np.array([[1.0, 2.0]]),
        point_labels=np.array([1]),
    )
    assert points[0] is not None and points[2] is None

    boxes = run('boxes', box=np.array([0.0, 0.0, 3.0, 3.0]))
    assert boxes[0][1][0].tolist() == [2, 3]

    masks = run('masks', mask_input=np.zeros((1, 8, 8), dtype=np.float32))
    assert masks[2] is not None

    with pytest.raises(ValueError):
        predictor.predict(point_coords=np.array([[1.0, 1.0]]))

    predictor.reset_image()
    assert predictor._features is None
    assert predictor._is_image_set is False
    predictor.reset_predictor()
    assert predictor._is_batch is False


def test_predictor_batch_paths():
    predictor, _ = _predictor()
    with pytest.raises(RuntimeError):
        predictor.predict_batch()
    with pytest.raises(TypeError):
        predictor.set_image_batch(np.zeros((2, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(TypeError):
        predictor.set_image_batch(['nope'])

    images = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((6, 6, 3), dtype=np.uint8)]
    predictor.set_image_batch(images)
    assert predictor._is_batch
    masks, scores, low_res = predictor.predict_batch(
        point_coords_batch=[np.array([[1.0, 1.0]]), None],
        point_labels_batch=[np.array([1]), None],
        box_batch=[None, np.array([0.0, 0.0, 2.0, 2.0])],
        mask_input_batch=[None, None],
    )
    assert len(masks) == 2
    assert len(scores) == 2
    assert len(low_res) == 2
