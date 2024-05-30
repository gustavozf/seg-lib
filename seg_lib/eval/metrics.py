from math import sqrt

import numpy as np
from sklearn.metrics import jaccard_score

class Metrics:
    eps=np.finfo(np.double).eps

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tps = 0
        self.fps = 0
        self.tns = 0
        self.fns = 0
        self.ious = []
        self.maes = []
        self.dices = []
        self.wfms = []
        self.emes = []
    
    def step(self, pred, gt):
        self.ious.append(self.get_iou(pred, gt))
        self.dices.append(self.get_dice(pred, gt))
        self.maes.append(self.get_mae(pred, gt))

        bool_gt = gt.astype(bool)
        bool_pred = pred.astype(bool)
        self.wfms.append(self.get_f_beta_measure(bool_pred, bool_gt))
        self.emes.append(self.get_e_measure(bool_pred, bool_gt))
        del bool_pred, bool_gt

    def get_iou(self, pred, gt):
        y_pred_bool = pred.astype(bool)
        y_true_bool = gt.astype(bool)
        tp = np.logical_and(y_true_bool, y_pred_bool).sum()
        fp = np.logical_and(~y_true_bool, y_pred_bool).sum()
        fn = np.logical_and(y_true_bool, ~y_pred_bool).sum()

        if  (tp + fn + fp) == 0:
            return 1.0 if tp==1.0 else 0.0

        return tp / (tp + fn + fp)

    def get_mae(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(pred - gt))

    def get_dice(self, pred, gt):
        y_pred_bool = pred.astype(bool)
        y_true_bool = gt.astype(bool)

        tp = np.logical_and( y_true_bool,  y_pred_bool).sum()
        fp = np.logical_and(~y_true_bool,  y_pred_bool).sum()
        fn = np.logical_and( y_true_bool, ~y_pred_bool).sum()
        
        if (tp + fn + fp) == 0:
            return 1.0 if tp == 0 else 0.0
        
        return 2 * tp / (2*tp + fn + fp)

    def cal_confusion(self, pred, gt):
        return (
            np.sum(pred[gt]==1),  # tp
            np.sum(pred[~gt]==1), # fp
            np.sum(pred[~gt]==0), # tn
            np.sum(pred[gt]==0)   # fn
        )

    def get_f_beta_measure(self, pred, gt, beta=sqrt(0.3)):
        tp, fp, _, fn = self.cal_confusion(pred, gt)

        if (tp + fn + fn) == 0:
            return 1.0 if tp == 0 else 0.0
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        beta_sqrd = beta**2
        return (
            ((beta_sqrd + 1) * precision * recall)
                / (beta_sqrd * precision + recall + 1e-8)
        )
    
    def get_e_measure(self, pred, gt):
        float_gt = gt.astype(np.float64)
        float_pred = pred.astype(np.float64)

        if np.sum(gt) == 0: #completely black
            enhanced_matrix = 1 - float_pred
        elif np.sum(~gt) == 0:
            enhanced_matrix = float_pred
        else:
            align_matrix =self.__e_meas_alignment_term(float_gt, float_pred)
            # calculate the enhanced alignment term
            enhanced_matrix=((align_matrix + 1)**2) / 4
        
        rows, cols = gt.shape
        return np.sum(enhanced_matrix)/(rows*cols+self.eps)
    
    def __e_meas_alignment_term(self, gt, pred):
        mean_gt = np.mean(gt)
        mean_pred = np.mean(pred)

        align_pred = pred - mean_pred
        align_gt = gt-mean_gt
        align_matrix = (
            2 * (align_gt * align_pred)
                / (align_gt**2 + align_pred**2 + self.eps)
        )
        return align_matrix

    def get_results(self) -> dict:
        return {
            'iou': np.array(self.ious).mean(),
            'dice': np.array(self.dices).mean(),
            'mae': np.array(self.maes).mean(),
            'f-measure': np.array(self.wfms).mean(),
            'e-measure': np.array(self.emes).mean()
        }

METRICS_PER_DATASET = {'common': Metrics}

def select_metric_from_dataset(dataset_name: str):
    if dataset_name in {'locuste', 'skin'}:
        return METRICS_PER_DATASET[dataset_name]
    
    return METRICS_PER_DATASET['common']