import cv2
import numpy as np
from seg_lib.eval.metrics import Metrics
from seg_lib.io.image import img_b64_to_arr

def get_interior_click(mask: np.ndarray):
    '''Returns the (x, y) pixel farthest from the mask border, which always
       lies inside the object (unlike the median of the coordinates).'''
    dist = cv2.distanceTransform(
        (mask > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return np.array([(x, y)])

class SamOracleInference:
    input_color = 'RGB'
    eval_mode = 'ORACLE'

    def __init__(self, predictor):
        self.predictor = predictor
        self.th = predictor.mask_threshold
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
        return get_interior_click(mask)

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
        return img_b64_to_arr(img_data, mode='RGB')

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
        h, w = img_size
        bbox = bbox[0] + np.random.randint(-8, 9, 4)
        xs = np.sort(np.clip(bbox[[0, 2]], 0, w - 1))
        ys = np.sort(np.clip(bbox[[1, 3]], 0, h - 1))

        return np.array([[xs[0], ys[0], xs[1], ys[1]]])

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
