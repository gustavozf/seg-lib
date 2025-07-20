import numpy as np

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
        self.device = device

    def predict(self, sam_inputs: list):
        if sam_inputs is None or len(sam_inputs) == 0:
            return None

        masks_local = []
        scores_local = []
        for inp in sam_inputs:
            masks, scores, _ = self.predictor.predict(
                **inp,
                multimask_output=True,
                return_logits=False
            )
            masks_local.append(masks)
            scores_local.append(scores)

        out_size = (len(sam_inputs), *self.predictor.original_size)
        return self.unify_mask(out_size, np.array(masks_local), np.array(scores_local))

    def segment_bbox(self, boxes: list = None):
        sam_input = [
            {
                "point_coords": None,
                "point_labels": None,
                "box": bbox
            }
            for bbox in boxes
        ]
        return self.predict(sam_input)

    def segment_point(self, points: list = None):
        sam_input = [
            {
                "point_coords": np.array(point[None, ...]),
                "point_labels": np.ones((1,), dtype=np.int64),
                "box": None
            }
            for point in points
        ]
        return self.predict(sam_input)
