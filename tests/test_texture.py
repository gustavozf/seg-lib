import numpy as np
import pytest

from seg_lib.texture.lpq import compute_response, lpq


def test_lpq_rejects_bad_arguments_and_returns_each_mode():
    image = np.arange(64, dtype=np.float64).reshape(8, 8)
    with pytest.raises(ValueError):
        lpq(image[None, ...])
    with pytest.raises(ValueError):
        lpq(image, winSize=4)
    with pytest.raises(ValueError):
        lpq(image, decorr=2)
    with pytest.raises(ValueError):
        lpq(image, freqestim=4)
    with pytest.raises(ValueError):
        lpq(image, mode='bad')

    response = compute_response(image, np.ones(3), np.ones(3))
    assert response.ndim == 2

    for freqestim in (1, 2, 3):
        for decorr in (0, 1):
            normalized = lpq(image, decorr=decorr, freqestim=freqestim, mode='nh')
            histogram = lpq(image, decorr=decorr, freqestim=freqestim, mode='h')
            codes = lpq(image, decorr=decorr, freqestim=freqestim, mode='im')
            assert normalized.ndim == 1 and normalized.shape[0] == 256
            assert normalized.sum() == pytest.approx(1.0)
            assert histogram.shape == normalized.shape
            assert histogram.sum() > 1
            assert codes.ndim == 2
