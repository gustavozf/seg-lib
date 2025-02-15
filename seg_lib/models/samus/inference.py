from seg_lib.models.selector import predictor_selector
from seg_lib.models.sam.inference import SamInference

class SamusInference(SamInference):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.predictor = predictor_selector(
            model_path,
            model_topology='SAMUS',
            model_type='vit_b',
            device=device)
        self.th = self.predictor.mask_threshold
