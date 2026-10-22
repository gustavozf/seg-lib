import torch

from seg_lib.models.device import get_device


def test_get_device_prefers_mps_then_cuda_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert get_device() == 'mps'

    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert get_device() == 'cuda'

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert get_device() == 'cpu'
