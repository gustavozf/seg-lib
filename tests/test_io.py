import base64
import os

import cv2
import numpy as np
import pytest

from seg_lib.io.files import dump_json, get_sorted_paths, load_data_paths, read_json
from seg_lib.io.image import (
    img_b64_to_arr,
    read_bmask,
    read_img,
    read_rmask,
)
from seg_lib.io.labelme import BoxPointFormatter, PointFormatter, SegPointFormatter
from seg_lib.models.checkpoint import strip_module_prefix


def test_json_round_trip_and_sorted_names(tmp_path):
    path = tmp_path / 'sample.json'
    dump_json(str(path), {'name': 'café', 'n': 2})
    assert read_json(str(path)) == {'name': 'café', 'n': 2}

    names = [
        str(tmp_path / '10.png'),
        str(tmp_path / '2.png'),
        str(tmp_path / '1.png'),
    ]
    ordered = get_sorted_paths(names)
    assert [os.path.basename(item) for item in ordered] == ['1.png', '2.png', '10.png']


def _touch(path, array, color=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, array)


def test_load_data_paths_missing_copied_and_unbalanced(tmp_path):
    assert load_data_paths(str(tmp_path / 'absent'), 'deeplab') == []

    root = tmp_path / 'ds'
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    gray = np.zeros((4, 4), dtype=np.uint8)
    _touch(str(root / 'imgs' / '1.png'), image)
    _touch(str(root / 'gt' / '1.png'), gray)
    _touch(str(root / 'segmentator_net' / '1.bmp'), gray)
    rows = load_data_paths(str(root), 'net')
    assert len(rows) == 1
    assert rows[0][2] == rows[0][3]

    _touch(str(root / 'imgs' / '2.png'), image)
    with pytest.raises(ValueError, match='Unbalanced'):
        load_data_paths(str(root), 'net')


def test_image_readers_and_labelme_formatters(tmp_path):
    color = np.zeros((5, 6, 3), dtype=np.uint8)
    color[:, :, 2] = 255
    gray = np.full((5, 6), 255, dtype=np.uint8)
    color_path = tmp_path / 'color.png'
    gray_path = tmp_path / 'gray.png'
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(gray_path), gray)

    rgb = read_img(str(color_path))
    assert rgb.shape == (5, 6, 3)
    assert rgb[:, :, 0].max() == 255
    assert read_bmask(str(gray_path)).max() == pytest.approx(1.0)
    assert read_rmask(str(color_path)).shape[0] == 5

    with pytest.raises(Exception):
        read_img(str(tmp_path / 'missing.png'))
    with pytest.raises(Exception):
        read_bmask(str(tmp_path / 'missing.png'))

    ok, encoded = cv2.imencode('.png', gray)
    assert ok
    decoded = img_b64_to_arr(base64.b64encode(encoded.tobytes()))
    assert decoded.shape[:2] == (5, 6)
    assert img_b64_to_arr(
        base64.b64encode(encoded.tobytes()), mode='RGB'
    ).shape == (5, 6, 3)

    rgba = np.zeros((5, 6, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    ok, encoded_rgba = cv2.imencode('.png', rgba)
    assert ok
    b64_rgba = base64.b64encode(encoded_rgba.tobytes())
    assert img_b64_to_arr(b64_rgba).shape == (5, 6, 4)
    assert img_b64_to_arr(b64_rgba, mode='RGB').shape == (5, 6, 3)

    with pytest.raises(NotImplementedError):
        PointFormatter.__call__([[0, 0]], 10, 10)
    seg = SegPointFormatter.__call__([[10, 20], [30, 40]], h=100, w=50)
    assert seg == ['0.2', '0.2', '0.6', '0.4']
    box = BoxPointFormatter.__call__([[0, 0], [10, 20]], h=100, w=50)
    assert len(box) == 4
    assert box[2] == '0.2'
    assert box[3] == '0.2'


def test_strip_module_prefix_only_touches_leading_prefix():
    state = {'module.a.weight': 1, 'b.module.bias': 2, 'c': 3}
    assert strip_module_prefix(state) == {
        'a.weight': 1, 'b.module.bias': 2, 'c': 3
    }
