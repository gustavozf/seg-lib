import logging

import numpy as np
import pandas as pd
import pytest

from seg_lib.eval.metrics import Metrics, select_metric_from_dataset
from seg_lib.eval.outputs import dump_metrics, dump_stats, print_metrics


def test_iou_and_dice_empty_overlap_and_errors():
    metrics = Metrics()
    empty = np.zeros((4, 4), dtype=bool)
    assert metrics.get_iou(empty, empty) == 1.0
    assert metrics.get_dice(empty, empty) == 1.0

    one = np.zeros((4, 4), dtype=bool)
    one[0, 0] = True
    assert metrics.get_iou(one, one) == 1.0
    assert metrics.get_dice(one, one) == 1.0
    assert metrics.get_iou(one, empty) == 0.0
    assert metrics.get_iou(empty, one) == 0.0

    overlap_pred = np.zeros((4, 4), dtype=bool)
    overlap_gt = np.zeros((4, 4), dtype=bool)
    overlap_pred[:, 0] = True
    overlap_gt[:, :2] = True
    assert metrics.get_iou(overlap_pred, overlap_gt) == pytest.approx(0.5)
    assert metrics.get_dice(overlap_pred, overlap_gt) == pytest.approx(2 / 3)


def test_f_measure_empty_false_positive_and_overlap():
    metrics = Metrics()
    empty = np.zeros((4, 4), dtype=bool)
    foreground = np.zeros((4, 4), dtype=bool)
    foreground[0, 0] = True
    assert metrics.get_f_beta_measure(empty, empty) == 1.0
    assert metrics.get_f_beta_measure(foreground, empty) == 0.0

    both = np.ones((4, 4), dtype=bool)
    score = metrics.get_f_beta_measure(both, foreground)
    assert 0.0 < score < 1.0


def test_e_measure_background_foreground_and_mixed():
    metrics = Metrics()
    empty = np.zeros((4, 4), dtype=bool)
    full = np.ones((4, 4), dtype=bool)
    mixed_pred = np.zeros((4, 4), dtype=bool)
    mixed_pred[:, :2] = True

    background = metrics.get_e_measure(mixed_pred, empty)
    foreground = metrics.get_e_measure(mixed_pred, full)
    mixed = metrics.get_e_measure(mixed_pred, mixed_pred)
    assert background == pytest.approx(1 - mixed_pred.astype(float).mean())
    assert foreground == pytest.approx(mixed_pred.astype(float).mean())
    assert mixed == pytest.approx(1.0)


def test_step_reset_and_results():
    metrics = Metrics()
    pred = np.ones((2, 2), dtype=np.float64)
    gt = np.ones((2, 2), dtype=np.float64)
    metrics.step(pred, gt)
    metrics.step(pred, np.zeros((2, 2), dtype=np.float64))
    results = metrics.get_results()
    assert len(metrics.ious) == 2
    assert set(results) == {'iou', 'dice', 'mae', 'f-measure', 'e-measure'}
    assert metrics.get_mae(pred.astype(float), np.zeros((2, 2))) == 1.0

    metrics.reset()
    assert metrics.ious == []
    assert metrics.tps == 0


def test_select_metric_falls_back_to_common():
    assert select_metric_from_dataset('common') is Metrics
    assert select_metric_from_dataset('skin') is Metrics
    assert select_metric_from_dataset('locuste') is Metrics


def test_print_metrics_scales_every_score_except_mae(caplog):
    metrics = Metrics()
    metrics.ious = [0.5]
    metrics.dices = [0.25]
    metrics.maes = [0.2]
    metrics.wfms = [0.1]
    metrics.emes = [0.4]
    with caplog.at_level(logging.INFO):
        print_metrics(metrics)
    text = caplog.text
    assert '50.00' in text
    assert '25.00' in text
    assert '10.00' in text
    assert '40.00' in text
    assert '0.20' in text
    assert '20.00' not in text


def test_dump_metrics_and_stats(tmp_path):
    keys = ['iou', 'dice', 'mae', 'f-measure', 'e-measure']
    baseline = {key: 0.1 for key in keys}
    predicted = {key: 0.2 for key in keys}
    fused = {'min': {key: 0.3 for key in keys}}
    metrics_path = tmp_path / 'metrics.csv'
    dump_metrics(str(metrics_path), baseline, predicted, fused)
    frame = pd.read_csv(metrics_path, index_col=0)
    assert list(frame.index) == keys
    assert frame.loc['iou', 'baseline'] == pytest.approx(0.1)
    assert frame.loc['dice', 'sam_pred'] == pytest.approx(0.2)
    assert frame.loc['mae', 'fusion_min'] == pytest.approx(0.3)

    stats_path = tmp_path / 'stats.csv'
    metrics = Metrics()
    metrics.ious = [0.4]
    metrics.dices = [0.5]
    metrics.maes = [0.6]
    metrics.wfms = [0.7]
    metrics.emes = [0.8]
    dump_stats(str(stats_path), ['a.png'], metrics)
    stats = pd.read_csv(stats_path)
    assert stats.loc[0, 'filenames'] == 'a.png'
    assert stats.loc[0, 'iou'] == pytest.approx(0.4)
