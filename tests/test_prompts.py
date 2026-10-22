import numpy as np
import pytest
import torch

from seg_lib.prompt.inference_after_sample import MaskInferenceAfterSample
from seg_lib.prompt.sampler import Sampler
from seg_lib.prompt.samus_sampler import (
    fixed_bbox,
    fixed_click,
    generate_click_prompt,
    pos_neg_clicks,
    random_bbox,
    random_click,
    random_clicks,
)
from seg_lib.prompt.train_sampler import TrainPromptSampler


def _blob(size=24, value=1):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[4:20, 4:20] = value
    return mask


def test_sampler_modes_fallback_and_grid_erosion():
    with pytest.raises(ValueError):
        Sampler(mode='missing')

    mask = _blob()
    blobs = (mask > 0).astype(np.int32)
    for mode in ('common', 'A', 'center_of_mass', 'B', 'random', 'C', 'grid', 'D'):
        points, labels = Sampler(mode=mode, min_blob_count=5, sampling_step=4).sample(blobs, mask)
        assert len(points) == len(labels)
        assert len(points) > 0

    tiny = np.zeros((12, 12), dtype=np.uint8)
    tiny[0:2, 0:2] = 1
    tiny_blobs = (tiny > 0).astype(np.int32)
    points, labels = Sampler(mode='common', min_blob_count=10).sample(tiny_blobs, tiny)
    assert labels.tolist() == [1]
    assert points[0].tolist() == [6, 6]

    empty = np.zeros((12, 12), dtype=np.uint8)
    points, labels = Sampler(mode='A', min_blob_count=1).sample(empty, empty)
    assert points[0].tolist() == [6, 6]

    eroded, labels = Sampler(
        mode='grid', erode_grid=True, min_blob_count=5, sampling_step=4
    ).sample(blobs, mask)
    plain, _ = Sampler(
        mode='grid', erode_grid=False, min_blob_count=5, sampling_step=4
    ).sample(blobs, mask)
    assert len(eroded) > 0
    assert len(plain) > 0
    assert labels.min() == 1


def test_train_prompt_sampler_branches():
    sampler = TrainPromptSampler(min_blob_count=10, max_num_prompts=2)
    with pytest.raises(ValueError):
        sampler.index_sampler_selector('nope')

    empty = np.zeros((16, 16), dtype=np.uint8)
    points, labels = sampler(empty.copy(), class_id=1, index_sampler='fixed')
    assert set(labels.tolist()) == {0}
    assert points.shape == (2, 2)

    small = np.zeros((16, 16), dtype=np.uint8)
    small[0:2, 0:2] = 1
    _, labels = sampler(small.copy(), class_id=1, index_sampler='random')
    assert set(labels.tolist()) == {0}

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[2:18, 2:10] = 1
    mask[2:20, 20:30] = 1
    one = TrainPromptSampler(min_blob_count=10, max_num_prompts=1)
    points, labels = one(mask.copy(), class_id=1, index_sampler='fixed')
    assert points.shape == (1, 2)
    assert labels.tolist() == [1]

    two = TrainPromptSampler(min_blob_count=10, max_num_prompts=2)
    points, labels = two(mask.copy(), class_id=1, index_sampler='fixed')
    assert points.shape == (2, 2)
    assert set(labels.tolist()) == {1}


def test_samus_prompt_helpers_cover_empty_masks():
    mask = _blob()
    empty = np.zeros_like(mask)
    point, label = fixed_click(mask, class_id=1)
    assert label == [1]
    assert point.shape == (1, 2)
    _, empty_label = fixed_click(empty, class_id=1)
    assert empty_label == [0]

    _, random_label = random_click(mask)
    _, random_empty = random_click(empty)
    assert random_label == [1]
    assert random_empty == [0]

    clicks, click_labels = random_clicks(mask, prompts_number=4)
    assert clicks.shape == (4, 2)
    assert set(click_labels.tolist()) == {1}
    _, empty_clicks = random_clicks(empty, prompts_number=3)
    assert set(empty_clicks.tolist()) == {0}

    both, both_labels = pos_neg_clicks(mask, pos_prompt_number=2, neg_prompt_number=3)
    assert both.shape == (5, 2)
    assert sorted(both_labels.tolist()) == [0, 0, 0, 1, 1]

    assert fixed_bbox(empty, img_size=24).tolist() == [-1, -1, 24, 24]
    assert random_bbox(empty, img_size=24).tolist() == [-1, -1, 24, 24]
    box = fixed_bbox(mask, img_size=24)
    assert box[2] > box[0] and box[3] > box[1]
    jitter = random_bbox(mask, img_size=24)
    assert jitter.shape == (4,)

    volume = torch.zeros(1, 1, 8, 8, 2)
    volume[0, 0, 2:5, 2:5, 0] = 1
    image = torch.zeros(1, 1, 8, 8, 2)
    _, points, prompt_mask = generate_click_prompt(image, volume)
    assert points.shape[0] == 1
    assert prompt_mask.shape[0] == 1
    blank = torch.zeros_like(volume)
    _, _, blank_mask = generate_click_prompt(image, blank)
    assert blank_mask.shape == prompt_mask.shape


def test_inference_after_sample_optional_samplers():
    class Predictor:
        mask_threshold = 0.0

        def __init__(self):
            self.model = self
            self.kwargs = None

        def set_image(self, image):
            self.image = image

        def predict(self, **kwargs):
            self.kwargs = kwargs
            masks = np.zeros((2, 4, 4))
            masks[1] = 1
            return masks, np.array([0.1, 0.9]), None

    predictor = Predictor()
    runner = MaskInferenceAfterSample(predictor, None, None, None)
    mask = runner(np.zeros((4, 4, 3)), np.zeros((4, 4)))
    assert predictor.kwargs['point_coords'] is None
    assert predictor.kwargs['box'] is None
    assert predictor.kwargs['mask_input'] is None
    assert mask.shape == (4, 4)

    def points(label):
        return np.array([[1, 1]]), np.array([1])

    runner = MaskInferenceAfterSample(
        predictor, points, lambda label: np.array([0, 0, 2, 2]), lambda label: np.ones((1, 4, 4))
    )
    runner(np.zeros((4, 4, 3)), np.ones((4, 4)))
    assert predictor.kwargs['box'] is not None
    assert predictor.kwargs['mask_input'] is not None
