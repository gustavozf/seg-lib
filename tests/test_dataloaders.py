import os

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from seg_lib.dataloaders.data_aug import (
    BaseAugmenter,
    NormalImgSegAugmenter,
    WeakImgSegAugmenter,
)
from seg_lib.dataloaders.image_ops import (
    resize_w_pad, to_grayscale, unpad_resize
)
from seg_lib.dataloaders.sam_dataset import SamGeneralDataset
from seg_lib.dataloaders.samus_data_aug import SamusImgAugmenter
from seg_lib.dataloaders.seg_dataset import SegGeneralDataset


def _write_pair(root, name, mask_value):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[1:7, 2:10] = mask_value
    subset = os.path.join(root, 'set_a')
    os.makedirs(os.path.join(subset, 'img'), exist_ok=True)
    os.makedirs(os.path.join(subset, 'label'), exist_ok=True)
    cv2.imwrite(os.path.join(subset, 'img', name), image)
    cv2.imwrite(os.path.join(subset, 'label', name), mask)


def _metadata(root, rows):
    meta = os.path.join(root, 'metadata')
    os.makedirs(meta, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(meta, 'dataset.csv'), index=False)


def test_seg_dataset_branches(tmp_path):
    _write_pair(tmp_path, 'a.png', 1)
    _write_pair(tmp_path, 'b.png', 2)
    _write_pair(tmp_path, 'missing-label.png', 1)
    os.remove(os.path.join(tmp_path, 'set_a', 'label', 'missing-label.png'))
    _metadata(tmp_path, [
        {'split': 'train', 'class_id': 1, 'subset': 'set_a', 'img_name': 'a.png', 'label_name': 'a.png'},
        {'split': 'train', 'class_id': 2, 'subset': 'set_a', 'img_name': 'b.png', 'label_name': 'b.png'},
        {'split': 'val', 'class_id': 1, 'subset': 'set_a', 'img_name': 'a.png', 'label_name': 'a.png'},
        {'split': 'train', 'class_id': 1, 'subset': 'set_a', 'img_name': 'gone.png', 'label_name': 'a.png'},
        {'split': 'train', 'class_id': 1, 'subset': 'set_a', 'img_name': 'missing-label.png', 'label_name': 'missing-label.png'},
    ])
    train = SegGeneralDataset(str(tmp_path), split='train', img_size=16)
    assert len(train) == 4
    assert train.get_num_classes('set_a') == 3

    sample = train[0]
    assert sample['image'].shape[0] == 3
    assert sample['label'].dtype == torch.long
    assert set(sample['label'].unique().tolist()).issubset({0, 1})

    gray = SegGeneralDataset(
        str(tmp_path), split='val', img_size=16, read_img_as_grayscale=True
    )
    gray_sample = gray[0]
    restored = gray_sample['image'] * gray.pixel_std + gray.pixel_mean
    assert torch.allclose(restored[0], restored[1], atol=1e-4)

    flagged = {'called': False}

    class FlagAug:
        def __call__(self, image, mask):
            flagged['called'] = True
            return image, mask

    augmented = SegGeneralDataset(
        str(tmp_path), split='val', img_size=16, augmenter=FlagAug()
    )
    augmented[0]
    assert flagged['called']

    plain = SegGeneralDataset(str(tmp_path), split='val', img_size=16, augmenter=None)
    assert 'image' in plain[0]

    with pytest.raises(ValueError, match='Image does not exist'):
        train[2]
    with pytest.raises(ValueError, match='Label does not exist'):
        train[3]


def test_binary_mask_is_thresholded_when_only_two_classes(tmp_path):
    _write_pair(tmp_path, 'a.png', 9)
    _metadata(tmp_path, [
        {'split': 'train', 'class_id': 1, 'subset': 'set_a', 'img_name': 'a.png', 'label_name': 'a.png'},
    ])
    dataset = SegGeneralDataset(str(tmp_path), img_size=16)
    assert dataset.get_num_classes('set_a') == 2
    label = dataset[0]['label']
    assert label.max() == 1


def test_sam_dataset_prompt_modes(tmp_path):
    _write_pair(tmp_path, 'a.png', 1)
    _metadata(tmp_path, [
        {'split': 'train', 'class_id': 1, 'subset': 'set_a', 'img_name': 'a.png', 'label_name': 'a.png'},
        {'split': 'val', 'class_id': 1, 'subset': 'set_a', 'img_name': 'a.png', 'label_name': 'a.png'},
    ])
    from seg_lib.prompt.train_sampler import TrainPromptSampler
    sampler = TrainPromptSampler(min_blob_count=1, max_num_prompts=1)
    click = SamGeneralDataset(
        str(tmp_path), prompt='click', img_size=16, embedding_size=8, point_sampler=sampler
    )
    bbox = SamGeneralDataset(str(tmp_path), prompt='bbox', img_size=16, embedding_size=8)
    both = SamGeneralDataset(
        str(tmp_path), prompt='all', img_size=16, embedding_size=8, point_sampler=sampler
    )
    val_bbox = SamGeneralDataset(
        str(tmp_path), split='val', prompt='bbox', img_size=16, embedding_size=8
    )

    click_sample = click[0]
    assert 'point_coords' in click_sample and 'boxes' not in click_sample
    assert 'boxes' in bbox[0] and 'point_coords' not in bbox[0]
    assert 'boxes' in both[0] and 'point_coords' in both[0]
    assert click_sample['low_mask'].shape[-1] == 8
    assert val_bbox[0]['boxes'][0] >= 0


def test_resize_and_grayscale_shapes():
    square = np.zeros((6, 6, 3), dtype=np.uint8)
    wide = np.zeros((4, 8, 3), dtype=np.uint8)
    gray = np.zeros((4, 8), dtype=np.uint8)
    assert resize_w_pad(square, (5, 5)).shape == (5, 5, 3)
    assert resize_w_pad(wide, (6, 6)).shape == (6, 6, 3)
    assert resize_w_pad(gray, (6, 6)).shape == (6, 6)
    color = np.zeros((5, 5, 3), dtype=np.uint8)
    color[:, :, 0] = 30
    assert to_grayscale(color, return_three_channels=True).shape == (5, 5, 3)
    assert to_grayscale(color, return_three_channels=False).shape == (5, 5)


def test_albumentations_augmenters_can_be_forced_off_and_on():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[0, 0] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    base_image, base_mask = BaseAugmenter()(image, mask)
    assert base_image.shape[0] == 32
    assert base_mask.shape == mask.shape

    for factory in (WeakImgSegAugmenter, NormalImgSegAugmenter):
        aug = factory(img_size=32)
        for transform in aug.transformations:
            transform.p = 0
        off_image, off_mask = aug(image, mask)
        assert off_image.shape == image.shape
        assert off_mask.shape == mask.shape
        for transform in aug.transformations:
            transform.p = 1
        on_image, on_mask = aug(image, mask)
        assert on_image is not None
        assert on_mask is not None


def _samus_pair():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[0, :, 0] = 255
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, 0] = 1
    return image, mask


def test_samus_augmenter_flags(monkeypatch):
    image, mask = _samus_pair()
    idle = SamusImgAugmenter.init_eval(img_size=16)
    out_image, out_mask = idle(image, mask)
    assert isinstance(out_image, np.ndarray)
    assert isinstance(out_mask, np.ndarray)
    assert out_image.shape == (16, 16, 3)
    assert out_mask.shape == (16, 16)

    train = SamusImgAugmenter.init_train(img_size=16)
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    same_image, same_mask = train(image, mask)
    assert same_image.shape == (16, 16, 3)

    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    assert np.array_equal(train.random_gamma(image), image)
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    pil_image = torch_to_pil(image)
    pil_mask = torch_to_pil(mask)
    assert train.horizontal_flip(pil_image, pil_mask)[0].size == (16, 16)
    assert train.random_rotation(pil_image, pil_mask)[0].size == (16, 16)
    assert train.random_scale(pil_image, pil_mask)[0].size == (16, 16)
    assert train.random_gauss_noise(pil_image).size == (16, 16)
    assert train.random_contrast(pil_image).size == (16, 16)
    assert train.random_distortion(pil_image).size == (16, 16)
    assert train.random_affine_transform(pil_image, pil_mask)[0].size == (16, 16)

    active = SamusImgAugmenter(
        img_size=16,
        crop=(8, 8),
        p_flip=1,
        p_rota=1,
        p_scale=1,
        p_gaussn=1,
        p_contr=1,
        p_gama=1,
        p_distor=1,
        p_random_affine=1,
    )
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    gamma = active.random_gamma(image.copy())
    assert gamma.dtype == np.uint8
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    flipped, _ = active.horizontal_flip(pil_image, pil_mask)
    assert flipped.getpixel((15, 0))[0] == 255
    cropped, cropped_mask = active.random_crop(pil_image, pil_mask)
    assert cropped.size == (8, 8)
    assert cropped_mask.size == (8, 8)
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    rotated, _ = active.random_rotation(pil_image, pil_mask)
    scaled, _ = active.random_scale(pil_image, pil_mask)
    noisy = active.random_gauss_noise(torch_to_pil(image[:, :, 0]))
    contrast = active.random_contrast(pil_image)
    distorted = active.random_distortion(pil_image)
    affine, affine_mask = active.random_affine_transform(pil_image, pil_mask)
    assert rotated.size == (16, 16)
    assert scaled.size == (16, 16)
    assert noisy.size == (16, 16)
    assert contrast.size == (16, 16)
    assert distorted.size == (16, 16)
    assert affine.size == (16, 16)
    assert affine_mask.size == (16, 16)
    assert same_mask.shape == (16, 16)


def test_samus_augmenter_rgb_noise_and_affine_without_crop(monkeypatch):
    image, mask = _samus_pair()
    aug = SamusImgAugmenter(
        img_size=16, crop=None, p_gaussn=1, p_random_affine=1,
        color_jitter_params=None,
    )
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    noisy = aug.random_gauss_noise(torch_to_pil(image))
    assert np.array(noisy).shape == (16, 16, 3)

    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    out_image, out_mask = aug(image, mask)
    assert out_image.shape == (16, 16, 3)
    assert out_mask.shape == (16, 16)


def test_samus_augmenter_output_feeds_sam_dataset(tmp_path):
    _write_pair(tmp_path, 'a.png', 1)
    _metadata(tmp_path, [
        {'split': 'train', 'class_id': 1, 'subset': 'set_a', 'img_name': 'a.png', 'label_name': 'a.png'},
    ])
    dataset = SamGeneralDataset(
        str(tmp_path), prompt='bbox', img_size=16, embedding_size=8,
        augmenter=SamusImgAugmenter.init_eval(img_size=16),
    )
    sample = dataset[0]
    assert sample['image'].shape == (3, 16, 16)


def test_resize_w_pad_keeps_class_values():
    mask = np.zeros((6, 10), dtype=np.uint8)
    mask[:, :5] = 2
    resized = resize_w_pad(mask, (23, 23), interpolation=cv2.INTER_NEAREST)
    assert set(np.unique(resized).tolist()) == {0, 2}


@pytest.mark.parametrize('shape', [(6, 10), (10, 6), (8, 8)])
def test_unpad_resize_inverts_resize_w_pad(shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:4, 2:5] = 1
    padded = resize_w_pad(mask, (40, 40), interpolation=cv2.INTER_NEAREST)
    restored = unpad_resize(padded, shape, interpolation=cv2.INTER_NEAREST)
    assert restored.shape == shape
    assert np.array_equal(restored, mask)


def test_num_classes_uses_every_split(tmp_path):
    _write_pair(tmp_path, 'a.png', 1)
    _write_pair(tmp_path, 'b.png', 2)
    _metadata(tmp_path, [
        {'split': 'train', 'class_id': 1, 'subset': 'set_a', 'img_name': 'a.png', 'label_name': 'a.png'},
        {'split': 'val', 'class_id': 2, 'subset': 'set_a', 'img_name': 'b.png', 'label_name': 'b.png'},
    ])
    train = SegGeneralDataset(str(tmp_path), split='train', img_size=16)
    val = SegGeneralDataset(str(tmp_path), split='val', img_size=16)
    assert train.get_num_classes('set_a') == val.get_num_classes('set_a') == 3

    # multi-class: class 1 is not merged with class 2
    label = train.read_label('b.png', 'set_a', 1)
    assert label.shape == (8, 12)
    assert label.sum() == 0
    assert val.read_label('b.png', 'set_a', 2).sum() == 6 * 8


def torch_to_pil(array):
    from torchvision.transforms import functional as F
    return F.to_pil_image(array)
