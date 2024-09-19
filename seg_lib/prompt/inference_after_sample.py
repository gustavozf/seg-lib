from collections import Callable

import numpy as np

class MaskInferenceAfterSample:
    def __init__(self,
            predictor,
            point_sampler: Callable,
            bbox_sampler: Callable,
            mask_sampler: Callable):
        self.predictor = predictor
        self.point_sampler = self.__set_point_sampler(point_sampler)
        self.bbox_sampler = self.__set_sampler(bbox_sampler)
        self.mask_sampler = self.__set_sampler(mask_sampler)
        self.th = predictor.model.mask_threshold

    def __set_sampler(self, sampler):
        if sampler is None:
            return lambda _: None

        return sampler
    
    def __set_point_sampler(self, sampler):
        if sampler is None:
            return lambda _: None, None

        return sampler

    def __call__(self, img, label):
        point_coords, point_labels = self.point_sampler(label)
        box = self.bbox_sampler(label)
        mask_input = self.mask_sampler(label)

        self.predictor.set_image(img)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=True,
            return_logits=False
        )
        best_score_idx = np.argmax(scores)
        binary_mask = masks[best_score_idx] > self.th

        return binary_mask
