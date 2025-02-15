import torch

from seg_lib.models.selector import predictor_selector
from seg_lib.models.sam.inference import SamInference

class SamMed2dInference(SamInference):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.predictor = predictor_selector(
            model_path,
            model_topology='SAM-Med2D',
            model_type='vit_b',
            device=device)
        self.th = self.predictor.mask_threshold

    def segment_bbox(self, boxes: list = None):
        if boxes is None or len(boxes) == 0:
            return None
        
        boxes = torch.tensor(boxes, device=self.predictor.device)
        boxes = self.predictor.apply_boxes_torch(
            boxes, self.predictor.original_size, self.predictor.new_size
        )
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=True,
            return_logits=False
        )
        out_size = (len(boxes), *self.predictor.original_size)
        return self.unify_mask(out_size, masks, scores)

    def segment_point(self, points: list = None):
        if points is None or len(points) == 0:
            return None

        points = torch.tensor(points, device=self.predictor.device)
        points = points[:, None, ...] # adds an additional axis
        points = self.predictor.apply_coords_torch(
            points, self.predictor.original_size, self.predictor.new_size
        )
        labels = [[1] for _ in range(len(points))]
        labels = torch.tensor(labels, device=self.predictor.device)

        masks, scores, _ = self.predictor.predict_torch(
            point_coords=points,
            point_labels=labels,
            boxes=None,
            multimask_output=True,
            return_logits=False
        )
        out_size = (len(points), *self.predictor.original_size)
        return self.unify_mask(out_size, masks, scores)
