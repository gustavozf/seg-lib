import cv2
import numpy as np
from seg_lib.eval.metrics import Metrics
from seg_lib.io.image import img_b64_to_arr

class SamOracleInference:
    input_color = 'RGB'
    eval_mode = 'ORACLE'

    def __init__(self, predictor):
        self.predictor = predictor
        self.th = 0.0 # predictor.model.mask_threshold
        self.metrics = {
            'click': Metrics(),
            'bbox': Metrics(),
            'comb': Metrics()
        }
        self.device = predictor.model.device

    def get_bbox(self, points: np.ndarray):
        min_x, min_y = points.min(axis=0).astype(int)
        max_x, max_y = points.max(axis=0).astype(int)
        return np.array([[min_x, min_y, max_x, max_y]])

    def get_clicks(self, mask: np.ndarray):
        y_indices, x_indices = np.nonzero(mask)
        n = len(x_indices) // 2

        # Compute the centroid
        centroid_x = sorted(x_indices)[n]
        centroid_y = sorted(y_indices)[n]
        return np.array([(centroid_x, centroid_y)])        

    def predict(self, pt_c, pt_l, bbox):
        masks, scores, _ = self.predictor.predict(
            point_coords=pt_c,
            point_labels=pt_l,
            box=bbox,
            mask_input=None,
            multimask_output=True,
            return_logits=False
        )
        best_score_idx = np.argmax(scores)
        return masks[best_score_idx] > self.th

    def reset(self):
        self.metrics['click'].reset()
        self.metrics['bbox'].reset()
        self.metrics['comb'].reset()

    def get_results(self):
        return { _k: self.metrics[_k].get_results() for _k in self.metrics }

    def load_img(self, img_data):
        return img_b64_to_arr(img_data)

    def __call__(self, data):
        base_mask = np.zeros((data['imageHeight'], data['imageWidth']))
        self.predictor.set_image(self.load_img(data['imageData']))

        for shape in data['shapes']:
            pts = np.array(shape['points'], dtype=np.int32)
            gt = cv2.fillPoly(base_mask.copy(), [pts], 1)
            bbox = self.get_bbox(pts)
            pt_c = self.get_clicks(gt)
            pt_l = np.ones((1), dtype=int)

            click_mask = self.predict(pt_c, pt_l, None)
            bbox_mask = self.predict(None, None, bbox)
            comb_mask = self.predict(pt_c, pt_l, bbox)

            self.metrics['click'].step(click_mask, gt)
            self.metrics['bbox'].step(bbox_mask, gt)
            self.metrics['comb'].step(comb_mask, gt)

class SamRandomInference(SamOracleInference):
    eval_mode = 'RANDOM'

    def randomiz_bbox(self, bbox: np.ndarray, img_size: tuple[int]):
        bbox = bbox + np.random.randint(-8, 8, 4)
        bbox = np.array([[
            np.clip(0, img_size[1], bbox[0][0]),
            np.clip(0, img_size[0], bbox[0][1]),
            np.clip(0, img_size[1], bbox[0][2]),
            np.clip(0, img_size[0], bbox[0][3])
        ]])

        return bbox

    def get_clicks(self, mask: np.ndarray):
        idxes = np.argwhere(mask == 1)[:, [1,0]]
        return np.array([idxes[np.random.randint(idxes.shape[0])]])
    
    def __call__(self, data):
        img_size = (data['imageHeight'], data['imageWidth'])
        base_mask = np.zeros(img_size)
        self.predictor.set_image(self.load_img(data['imageData']))

        for shape in data['shapes']:
            pts = np.array(shape['points'], dtype=np.int32)
            gt = cv2.fillPoly(base_mask.copy(), [pts], 1)

            bbox = self.get_bbox(pts)
            bbox = self.randomiz_bbox(bbox, img_size)
            pt_c = self.get_clicks(gt)
            pt_l = np.ones((1), dtype=int)

            click_mask = self.predict(pt_c, pt_l, None)
            bbox_mask = self.predict(None, None, bbox)
            comb_mask = self.predict(pt_c, pt_l, bbox)

            self.metrics['click'].step(click_mask, gt)
            self.metrics['bbox'].step(bbox_mask, gt)
            self.metrics['comb'].step(comb_mask, gt)