import torch

CLEAN_CACHE = {
    'mps': torch.mps.empty_cache,
    'cuda': torch.cuda.empty_cache,
    'cpu': lambda: None
}

def get_device():
  if torch.backends.mps.is_available():
    return "mps"
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"