import numpy as np
import torch

from tests.conftest import FakePredictor
from seg_lib.models.sam.inference import SamInference
from seg_lib.models.sam2.inference import Sam2Inference, Sam21Inference
from seg_lib.models.sam_med2d.inference import SamMed2dInference
from seg_lib.models.samus.inference import SamusInference
from seg_lib.models.samus.predictor import SamusPredictor, SamusResizeLongestSide


def _inference(cls):
    obj = cls.__new__(cls)
    obj.predictor = FakePredictor()
    obj.th = 0.0
    obj.device = 'cpu'
    return obj


def test_segment_prompt_combinations():
    inf = _inference(SamInference)
    calls = []
    inf.segment_point = lambda points: calls.append(('points', None if points is None else points.shape)) or np.ones((1, 2, 2))
    inf.segment_bbox = lambda boxes: calls.append(('boxes', None if boxes is None else boxes.shape)) or np.ones((1, 2, 2))

    boxes_only = inf.segment(points=None, boxes=[[0, 0, 1, 1]])
    assert calls == [('boxes', (1, 4))]
    assert boxes_only.shape == (1, 2, 2)

    calls.clear()
    points_only = inf.segment(points=[[1, 2]], boxes=None)
    assert calls == [('points', (1, 2))]
    assert points_only.shape == (1, 2, 2)

    calls.clear()
    both = inf.segment(points=[[1, 2]], boxes=[[0, 0, 1, 1]])
    assert [name for name, _ in calls] == ['points', 'boxes']
    assert both.shape == (2, 2, 2)

    calls.clear()
    assert inf.segment(points=None, boxes=None) is None
    assert calls == []


def test_empty_prompts_return_none_for_each_inference_class():
    for cls in (SamInference, Sam2Inference, Sam21Inference, SamMed2dInference, SamusInference):
        inf = _inference(cls)
        assert inf.segment_bbox(None) is None
        assert inf.segment_bbox([]) is None
        assert inf.segment_point(None) is None
        assert inf.segment_point([]) is None


def test_base_segment_methods_call_the_predictor():
    inf = _inference(SamInference)
    masks = inf.segment_bbox([[0, 0, 1, 1]])
    assert masks.shape == (1, 4, 4)
    assert masks[0].any()
    assert any(name == 'predict_torch' for name, *_ in inf.predictor.calls)

    point_masks = inf.segment_point([[1, 1]])
    assert point_masks.shape == (1, 4, 4)


def test_sam2_segment_uses_predict_api():
    inf = _inference(Sam2Inference)
    masks = inf.segment_bbox([[0, 0, 3, 3]])
    assert masks.shape == (1, 4, 4)
    recorded = inf.predictor.calls[-1]
    assert recorded[0] == 'predict'
    assert recorded[1]['box'].shape == (1, 4)

    point_masks = inf.segment_point([[1, 2]])
    assert point_masks.shape == (1, 4, 4)
    assert inf.predictor.calls[-1][1]['point_coords'].shape[0] == 1


def test_unify_mask_accepts_torch_numpy_and_single_mask():
    inf = _inference(SamInference)
    masks = torch.zeros(1, 3, 2, 2)
    masks[0, 2] = 1
    scores = torch.tensor([[0.1, 0.2, 0.7]])
    unified = inf.unify_mask((1, 2, 2), masks, scores)
    assert unified.dtype == bool
    assert unified[0].all()

    numpy_masks = np.zeros((2, 3, 2, 2))
    numpy_masks[:, 0] = 1
    numpy_scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]])
    unified = inf.unify_mask((2, 2, 2), numpy_masks, numpy_scores)
    assert unified[0].all()
    assert not unified[1].any()

    single_masks = np.zeros((3, 2, 2))
    single_masks[1] = 1
    single = inf.unify_mask((1, 2, 2), single_masks, np.array([0.1, 0.9, 0.0]))
    assert single.shape == (1, 2, 2)
    assert single[0].all()


class _SamusHost:
    mask_threshold = 0.0
    image_format = 'RGB'
    device = torch.device('cpu')


def test_samus_inference_reads_predictor_threshold(monkeypatch):
    import seg_lib.models.samus.inference as samus_inference

    monkeypatch.setattr(
        samus_inference, 'predictor_selector',
        lambda *args, **kwargs: SamusPredictor(_SamusHost()),
    )
    inf = SamusInference('w.pt')
    assert inf.th == 0.0
    assert inf.predictor.mask_threshold == 0.0


def test_samus_transform_matches_centered_padding():
    transform = SamusResizeLongestSide(256)
    # (h, w) = (100, 200): pad 50 rows on top, scale 256 / 200
    coords = np.array([[0.0, 0.0], [200.0, 100.0]])
    out = transform.apply_coords(coords, (100, 200))
    assert np.allclose(out, [[0.0, 64.0], [256.0, 192.0]])

    boxes = np.array([[0, 0, 200, 100]])
    assert np.allclose(transform.apply_boxes(boxes, (100, 200)), [[0, 64, 256, 192]])

    torch_out = transform.apply_coords_torch(torch.tensor(coords), (100, 200))
    assert torch.allclose(torch_out, torch.tensor(out, dtype=torch.float))

    # tall image: pad columns instead
    tall = transform.apply_coords(np.array([[0.0, 0.0]]), (200, 100))
    assert np.allclose(tall, [[64.0, 0.0]])

    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    resized = transform.apply_image(image)
    assert resized.shape == (256, 256, 3)
    assert resized[0, 128].sum() == 0
    assert resized[128, 128].sum() > 0


def test_samus_postprocess_removes_centered_padding():
    predictor = SamusPredictor(_SamusHost())
    predictor.original_size = (4, 8)
    low_res = torch.zeros(1, 1, 8, 8)
    low_res[..., 2:6, :] = 1
    masks = predictor.postprocess_masks(low_res)
    assert masks.shape == (1, 1, 4, 8)
    assert torch.all(masks == 1)
