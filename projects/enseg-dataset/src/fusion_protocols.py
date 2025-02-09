import gc

import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from seg_lib.io.files import read_json
from seg_lib.eval.metrics import Metrics
from seg_lib.io.image import img_b64_to_arr

class FusionProtocol1:
    '''
        Candidate-first: if `p` in `c`, consider `c` as the final output
        and `s` otherwise;
    '''
    def __init__(self, yolo, sam):
        self.yolo = yolo
        self.sam = sam
        self.metrics = Metrics()
        self.th = sam.model.mask_threshold

    def get_fixed_clicks(self, mask: np.ndarray):
        # idxes = np.argwhere(mask == 1)[:, [1,0]]
        # return np.array([idxes[idxes.shape[0]//2]])
        y_indices, x_indices = np.nonzero(mask)
        n = len(x_indices) // 2

        # Compute the centroid
        centroid_x = sorted(x_indices)[n]
        centroid_y = sorted(y_indices)[n]
        return np.array([(centroid_x, centroid_y)])        

    def get_random_clicks(self, mask: np.ndarray):
        idxes = np.argwhere(mask == 1)[:, [1,0]]
        return np.array([idxes[np.random.randint(idxes.shape[0])]])

    def load_imgs_from_labelme(self, labels_paths):
        imgs = []
        shapes = []
        print('Loading input images...')
        for label_path in labels_paths:
            data = read_json(label_path)
            imgs.append(img_b64_to_arr(data['imageData'])[..., [2, 1, 0]])
            shapes.append(data['shapes'])
            del data

        return imgs, shapes

    def predict_candidates(self, imgs):
        print('Generating YOLO predictions...')
        preds = self.yolo.predict(imgs, verbose=False)
        original_img_sizes = [pred.orig_shape for pred in preds]

        pred_xy = [
            [] if pred.masks is None else pred.masks.xy
            for pred in preds
        ]
        del preds

        return pred_xy, original_img_sizes

    def select_candidate(self, prompt, candidates):
        best_cand_id = None
        pt = Point(*prompt[0])
        for i in range(len(candidates)):
            if Polygon(candidates[i]).contains(pt):
                best_cand_id = i
                break

        return best_cand_id

    def predict_sam(self, pt_c = None, box = None):
        masks, scores, _ = self.sam.predict(
            point_coords=pt_c,
            point_labels=None if pt_c is None else np.ones(1, dtype=int),
            box=box,
            mask_input=None,
            multimask_output=True,
            return_logits=False
        )
        best_score_idx = np.argmax(scores)
        return masks[best_score_idx] > self.th

    def apply_protocol(self, prompt, best_cand_id, candts_pts, base_mask):
        '''
          Candidate-first: if `p` in `c`, consider `c` as the final output
            and `s` otherwise;
        '''
        if best_cand_id is None:
            return self.predict_sam(pt_c=np.array(prompt, dtype=np.int32))

        best_mask = candts_pts[best_cand_id]
        best_mask = cv2.fillPoly(base_mask.copy(), [best_mask.astype(int)], 1)
        del candts_pts[best_cand_id]
        return best_mask

    def __call__(self, labels_paths):
        self.metrics.reset()
        num_matches = 0
        num_gt_cells = 0
        num_pred_objs = 0
        imgs, gt_shapes = self.load_imgs_from_labelme(labels_paths)
        cndts_xy, original_img_sizes = self.predict_candidates(imgs)

        # for each image
        for i in tqdm(range(len(imgs))):
            self.sam.set_image(imgs[i])
            base_mask = np.zeros(original_img_sizes[i])
            cndts_xy[i] = [
                cndt_shape
                for cndt_shape in cndts_xy[i]
                if cndt_shape.sum() > 0
            ]
            num_pred_objs += len(cndts_xy[i])
            num_gt_cells += len(gt_shapes[i])

            # for each cell polygonon
            for pts in gt_shapes[i]:
                gt_mask = cv2.fillPoly(
                    base_mask.copy(),
                     [np.array(pts['points'], dtype=np.int32)],
                    1)
                prompt = self.get_fixed_clicks(gt_mask)

                # select the best candidate
                best_cand_id = self.select_candidate(prompt, cndts_xy[i])
                # apply the fusion protocol
                final_mask = self.apply_protocol(
                    prompt, best_cand_id, cndts_xy[i], base_mask
                )
                num_matches += int(best_cand_id is not None)
                self.metrics.step(final_mask, gt_mask)
            gc.collect()
        return {
            **self.metrics.get_results(),
            'num_matches': num_matches,
            'num_gt_cells': num_gt_cells,
            'num_pred_objs': num_pred_objs
        }
    
class FusionProtocol2(FusionProtocol1):
    def apply_protocol(self, prompt, best_cand_id, candts_pts, base_mask):
        '''
          Prompt refinement: if `p` in `c`, calculate a bounding box from `c`
            and regenerate `s`. Otherwise, `s` is the final output.
        '''
        if best_cand_id is None:
            return self.predict_sam(pt_c=prompt)

        best_mask = candts_pts[best_cand_id]
        best_mask = cv2.fillPoly(base_mask.copy(), [best_mask.astype(int)], 1)
        del candts_pts[best_cand_id]

        idxes = np.argwhere(best_mask == 1)[:, [1,0]]
        x_min, y_min = idxes.min(axis=0)
        x_max, y_max = idxes.max(axis=0)

        return self.predict_sam(box=np.array([[x_min, y_min, x_max, y_max]]))
    
class FusionProtocol3(FusionProtocol1):
    def apply_protocol(self, prompt, best_cand_id, candts_pts, base_mask):
        '''
          Segmentation fusion: if `p` in `c`, combine both `c` and `s` using a
            fusion rule. Otherwise, s is the final output
        '''
        sam_pred = self.predict_sam(pt_c=prompt)
        if best_cand_id is None:
            return sam_pred

        best_mask = candts_pts[best_cand_id]
        best_mask = cv2.fillPoly(base_mask.copy(), [best_mask.astype(int)], 1)
        del candts_pts[best_cand_id]

        return (best_mask.astype(bool) & sam_pred.astype(bool)).astype(int)
    
class FusionProtocol4(FusionProtocol1):
    def apply_protocol(self, prompt, best_cand_id, candts_pts, base_mask):
        '''
          Prompt refinement w/ segmentation fusion: similar to Protocol 2,
            but combining the outputted `c` and `s`
        '''
        sam_pred_pt = self.predict_sam(pt_c=prompt)
        if best_cand_id is None:
            return sam_pred_pt

        best_mask = candts_pts[best_cand_id]
        best_mask = cv2.fillPoly(base_mask.copy(), [best_mask.astype(int)], 1)
        del candts_pts[best_cand_id]

        idxes = np.argwhere(best_mask == 1)[:, [1,0]]
        x_min, y_min = idxes.min(axis=0)
        x_max, y_max = idxes.max(axis=0)
        bbox = np.array([[x_min, y_min, x_max, y_max]])
        sam_pred_bx = self.predict_sam(box=bbox)
        return (
            sam_pred_pt.astype(bool) & sam_pred_bx.astype(bool)
        ).astype(int)