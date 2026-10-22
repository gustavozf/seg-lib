import numpy as np
import pytest

from seg_lib.eval.fusion import KittlerLogitsFusion, LogitsFusion


def test_logits_fusion_rules_and_unknown_method():
    source = np.array([[0.2, 0.8]], dtype=np.float32)
    predicted = np.array([[0.6, 0.1]], dtype=np.float32)
    rules = {
        'default': (2 * source + predicted) / 3,
        'custom': (2 * source + np.minimum(source, predicted)) / 3,
        'min': np.minimum(source, predicted),
        'max': np.maximum(source, predicted),
        'mean': (source + predicted) / 2,
        'median': np.median([source, predicted], axis=0),
    }
    for name, expected in rules.items():
        got = LogitsFusion.apply(source, predicted, method=name)
        np.testing.assert_allclose(got, expected)

    np.testing.assert_allclose(
        LogitsFusion.apply(source, predicted, method=None),
        rules['default'],
    )
    with pytest.raises(ValueError):
        LogitsFusion.select_method('missing')


def test_kittler_fusion_extra_rules_and_probabilities():
    source = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    predicted = np.array([0.2, 0.4, 0.9], dtype=np.float32)
    probs = KittlerLogitsFusion.to_probs(source)
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs[:, 1], source)
    np.testing.assert_allclose(probs[:, 0], 1 - source)

    for method in ('sum', 'sub', 'prod', 'default'):
        classes = KittlerLogitsFusion.apply(source, predicted, method=method)
        assert classes.dtype == np.float32
        assert set(np.unique(classes)).issubset({0.0, 1.0})
