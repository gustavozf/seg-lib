import logging

import numpy as np
import torch

from seg_lib.models.selector import predictor_selector

logger = logging.getLogger()

class SamInference:
    def __init__(
            self,
            model_path: str,
            encoder_type: str = 'vit_b',
            device: str = 'cpu'):
        self.predictor = predictor_selector(
            model_path,
            model_topology='SAM',
            model_type=encoder_type,
            device=device)
        self.th = self.predictor.mask_threshold

    def clear(self):
        logger.debug("Resetting SAM image")
        self.predictor.reset_image()
    
    def set_image(self, image):
        logger.debug("Setting SAM image")
        self.predictor.set_image(image)

    def unify_mask(self, size, masks, scores):
        best_scores = np.argmax(scores, axis=1)
        unified_mask = np.zeros(size, dtype=bool)
        for i in range(len(scores)):
            unified_mask[i] = masks[i][best_scores[i]] > self.th

        return unified_mask

    def segment_bbox(self, boxes: list = None):
        if boxes is None or len(boxes) == 0:
            logger.debug("No boxes to segment")
            return None
        
        logger.debug("Segmenting Bounding Boxes")
        boxes = torch.tensor(boxes, device=self.predictor.device)
        boxes = self.predictor.transform.apply_boxes_torch(
            boxes, self.predictor.original_size
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
            logger.debug("No points to segment")
            return None

        logger.debug("Segmenting Points")
        points = torch.tensor(points, device=self.predictor.device)
        points = points[:, None, ...] # adds an additional axis
        points = self.predictor.transform.apply_coords_torch(
            points, self.predictor.original_size
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

    def segment(self, points: list = None, boxes: list = None):
        logger.debug("Running SAM inference")
        points_masks = self.segment_point(points)
        bbox_masks = self.segment_bbox(boxes)
        
        if points_masks is None and bbox_masks is None:
            return None
        if points_masks is None:
            return bbox_masks
        if bbox_masks is None:
            return points_masks
        return np.concatenate([points_masks, bbox_masks], axis=0)
    
    def __call__(self, image, points: list = None, boxes: list = None):
        self.set_image(image)
        return self.segment(points=points, boxes=boxes)