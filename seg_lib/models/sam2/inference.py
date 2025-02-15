import torch
import numpy as np

from seg_lib.models.selector import predictor_selector
from seg_lib.models.sam.inference import SamInference

class Sam2Inference(SamInference):
    def __init__(
            self,
            model_path: str,
            encoder_type: str = 'hiera_t',
            device: str = 'cpu'):
        self.predictor = predictor_selector(
            model_path,
            model_topology='SAMv2',
            model_type=encoder_type,
            device=device)
        self.th = self.predictor.mask_threshold

    def segment_bbox(self, boxes: list = None):
        if boxes is None or len(boxes) == 0:
            return None
        
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(boxes),
            multimask_output=True,
            return_logits=False
        )
        out_size = (len(boxes), *masks.shape[-2:])
        return self.unify_mask(out_size, masks, scores)
    
    def segment_point(self, points: list = None):
        if points is None or len(points) == 0:
            return None

        points = torch.tensor(points, device=self.predictor.device)
        points = points[:, None, ...] # adds an additional axis
        labels = [[1] for _ in range(len(points))]
        labels = torch.tensor(labels, device=self.predictor.device)

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=True,
            return_logits=False
        )
        out_size = (len(points), *masks.shape[-2:])
        return self.unify_mask(out_size, masks, scores)

class Sam21Inference(Sam2Inference):
    def __init__(
            self,
            model_path: str,
            encoder_type: str = 'hiera_t',
            device: str = 'cpu'):
        self.predictor = predictor_selector(
            model_path,
            model_topology='SAMv2.1',
            model_type=encoder_type,
            device=device)
        self.th = self.predictor.mask_threshold