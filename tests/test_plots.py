import torch

from seg_lib.eval.plots import plot_dataloader_sample


def _batch(with_points, with_boxes):
    batch = {
        'image': torch.zeros(2, 1, 8, 8),
        'label': torch.zeros(2, 1, 8, 8),
    }
    batch['image'][0, 0, 0, 0] = 1
    if with_points:
        batch['point_coords'] = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    if with_boxes:
        batch['boxes'] = torch.tensor([[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 4.0, 4.0]])
    return batch


def test_plot_sample_with_and_without_prompts(tmp_path):
    plain = plot_dataloader_sample(_batch(False, False), sample_size=2)
    assert plain is not None
    plain.clf()

    prompted = plot_dataloader_sample(_batch(True, True), sample_size=2)
    assert prompted.axes[0].lines
    assert prompted.axes[0].patches
    prompted.clf()

    missing_points = _batch(False, True)
    missing_points['point_coords'] = None
    missing_boxes = _batch(True, False)
    missing_boxes['boxes'] = None
    plot_dataloader_sample(missing_points, sample_size=2).clf()
    plot_dataloader_sample(missing_boxes, sample_size=2).clf()

    output = tmp_path / 'sample.png'
    plot_dataloader_sample(_batch(True, False), sample_size=2, output_path=str(output))
    assert output.exists()
    omitted = tmp_path / 'omitted.png'
    plot_dataloader_sample(_batch(False, False), sample_size=2, output_path=None)
    assert not omitted.exists()
