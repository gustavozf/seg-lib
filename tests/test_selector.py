import torch
import pytest

import seg_lib.models.selector as selector
from seg_lib.models.sam.predictor import SamPredictor
from seg_lib.models.sam2.predictor import Sam2Predictor
from seg_lib.models.sam_med2d.predictor import SamMed2DPredictor
from seg_lib.models.samus.predictor import SamusPredictor


def test_supported_model_types_keep_each_vit_name():
    assert selector.SUPPORTED_MODEL_TYPES == {'default', 'vit_h', 'vit_l', 'vit_b'}


def test_seg_selector_rejects_unknown_and_builds_known(monkeypatch):
    def explode(checkpoint):
        raise AssertionError(checkpoint)

    monkeypatch.setitem(selector.SEG_MODELS, 'SegPVT', explode)
    with pytest.raises(ValueError):
        selector.seg_selector('ckpt', model_topology='missing')

    built = []
    monkeypatch.setitem(
        selector.SEG_MODELS,
        'CAFE',
        lambda checkpoint: built.append(checkpoint) or torch.nn.Linear(1, 1),
    )
    model = selector.seg_selector('cafe.pt', model_topology='CAFE', device='cpu')
    assert built == ['cafe.pt']
    assert isinstance(model, torch.nn.Linear)


class _Host:
    mask_threshold = 0.0
    image_size = 16
    image_format = 'RGB'
    device = torch.device('cpu')

    def __init__(self):
        self.image_encoder = torch.nn.Identity()
        self.image_encoder.img_size = 16


def test_sam_selector_dispatches_each_topology(monkeypatch):
    calls = []

    def record(name):
        def build(*args, **kwargs):
            calls.append((name, args, kwargs))
            return _Host()
        return build

    monkeypatch.setattr(selector, 'build_sam_model', record('SAM'))
    monkeypatch.setattr(selector, 'build_samus_model', record('SAMUS'))
    monkeypatch.setattr(selector, 'build_sam_med_2d_model', record('SAM-Med2D'))
    monkeypatch.setattr(selector, 'build_samv2_model', record('SAMv2'))
    monkeypatch.setattr(selector, 'build_samv2_1_model', record('SAMv2.1'))

    with pytest.raises(ValueError):
        selector.sam_selector('w.pt', model_topology='nope')

    selector.sam_selector('a.pt', model_topology='SAM', model_type='vit_b', device='cpu')
    selector.sam_selector('b.pt', model_topology='SAMUS')
    selector.sam_selector('c.pt', model_topology='SAM-Med2D')
    selector.sam_selector('d.pt', model_topology='SAMv2', model_type='hiera_t')
    selector.sam_selector('e.pt', model_topology='SAMv2.1', model_type='hiera_s')
    assert [name for name, _, _ in calls] == [
        'SAM', 'SAMUS', 'SAM-Med2D', 'SAMv2', 'SAMv2.1'
    ]
    assert calls[3][2]['model_type'] == 'hiera_t'
    assert calls[4][2]['model_type'] == 'hiera_s'


def test_predictor_selector_wraps_the_matching_class(monkeypatch):
    monkeypatch.setattr(selector, 'sam_selector', lambda *args, **kwargs: _Host())
    with pytest.raises(ValueError):
        selector.predictor_selector('w.pt', model_topology='nope')

    assert isinstance(
        selector.predictor_selector('w.pt', model_topology='SAM'), SamPredictor
    )
    assert isinstance(
        selector.predictor_selector('w.pt', model_topology='SAMUS'), SamusPredictor
    )
    assert isinstance(
        selector.predictor_selector('w.pt', model_topology='SAM-Med2D'),
        SamMed2DPredictor,
    )
    assert isinstance(
        selector.predictor_selector('w.pt', model_topology='SAMv2'), Sam2Predictor
    )
    assert isinstance(
        selector.predictor_selector('w.pt', model_topology='SAMv2.1'), Sam2Predictor
    )
