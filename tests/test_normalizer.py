import os

import numpy as np
import pytest

from seg_lib.models.normalizer import Normalizer


def test_each_normalizer_method_and_overrides(tmp_path, monkeypatch):
    logits = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        Normalizer.apply(logits, method='missing')

    sigmoid = Normalizer.apply(logits, method='sigmoid')
    assert sigmoid.shape == logits.shape
    assert sigmoid.min() >= 0 and sigmoid.max() <= 1

    defaulted = Normalizer.min_max(logits)
    overridden = Normalizer.min_max(logits, min_v=0, max_v=6)
    assert defaulted.min() == pytest.approx(0)
    assert defaulted.max() == pytest.approx(1)
    assert overridden.max() == pytest.approx(0.5)

    z_default = Normalizer.z_score(logits)
    z_custom = Normalizer.z_score(logits, mean=0, stdv=1)
    assert z_default.mean() == pytest.approx(0, abs=1e-5)
    np.testing.assert_allclose(z_custom, logits)

    stretched = Normalizer.z_score_std(logits, mean=1, stdv=1, min_v=None, max_v=None)
    fixed = Normalizer.z_score_std(logits, mean=0, stdv=1, min_v=0, max_v=10)
    assert stretched.min() == pytest.approx(0)
    assert stretched.max() == pytest.approx(1)
    assert fixed.max() < 1

    raw = Normalizer.logit_norm(logits, t=1.0)
    scaled = Normalizer.logit_norm(logits, t=2.0)
    assert raw.shape == logits.shape
    assert np.max(np.abs(scaled)) < np.max(np.abs(raw)) + 1e-6
    unit = Normalizer.logit_norm_std(logits, t=1.0, min_v=None, max_v=None)
    bounded = Normalizer.logit_norm_std(logits, t=1.0, min_v=-1, max_v=1)
    assert unit.min() == pytest.approx(0)
    assert unit.max() == pytest.approx(1)
    assert bounded.max() <= 1

    created = {}

    def fake_mkstemp(prefix, suffix):
        path = tmp_path / f'{prefix}temp{suffix}'
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        created['path'] = path
        return fd, str(path)

    monkeypatch.setattr('seg_lib.models.normalizer.tempfile.mkstemp', fake_mkstemp)
    probabilities = Normalizer.cv_norm(logits, to_prob=True)
    assert not created['path'].exists()
    inverted = Normalizer.cv_norm(logits, to_prob=False)
    assert not created['path'].exists()
    assert probabilities.shape == logits.shape
    assert inverted.shape == logits.shape
    assert probabilities.max() <= 1
    assert inverted.dtype == np.uint8
