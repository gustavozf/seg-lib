import tempfile

import cv2
import torch
import numpy as np

class Normalizer:
    AVAILABLE_METHODS = {
        'sigmoid',
        'min_max',
        'z_score',
        'z_score_std',
        'logit_norm',
        'logit_norm_std',
        'cv_norm'
    }

    @classmethod
    def apply(cls, logits, method: str = 'logit_norm_std', **kwargs):
        method = cls.select_method(method=method)
        return method(logits, **kwargs)

    @classmethod
    def select_method(cls, method: str = 'sigmoid'):
        if method not in cls.AVAILABLE_METHODS:
            raise ValueError(
                f'Method {method} not supported! '
                f'Supported methods: {cls.AVAILABLE_METHODS}'
            )
        
        if method == 'logit_norm_std':
            return cls.logit_norm_std
        if method == 'min_max':
            return cls.min_max
        if method == 'z_score':
            return cls.z_score
        if method == 'z_score_std':
            return cls.z_score_std
        if method == 'logit_norm':
            return cls.logit_norm
        if method == 'cv_norm':
            return cls.cv_norm
        
        return cls.sigmoid

    @staticmethod
    def sigmoid(logits: np.ndarray) -> np.ndarray:
        return torch.sigmoid(torch.from_numpy(logits)).numpy()
    
    @staticmethod
    def min_max(
        logits: np.ndarray, min_v: int = None, max_v: int = None
    ) -> np.ndarray:
        if min_v is None:
            min_v = logits.min()
        if max_v is None:
            max_v = logits.max()

        return (logits - min_v) / (max_v - min_v)
    
    @staticmethod
    def z_score(
        logits: np.ndarray, mean: float = None, stdv: float = None
    ) -> np.ndarray:
        if mean is None:
            mean = logits.mean()
        if stdv is None:
            stdv = np.std(logits)
        
        return (logits - mean) / stdv
    
    @staticmethod
    def logit_norm(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
        '''
            Method described in the paper `Mitigating Neural Network
            Overconfidence with Logit Normalization`
            Link: https://arxiv.org/abs/2205.09310
            GitHub: https://github.com/hongxin001/logitnorm_ood/
            License: [NOT DECLARED]
        '''
        logits = torch.from_numpy(logits)
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        return (torch.div(logits, norms) / t).numpy()
    
    @classmethod
    def z_score_std(
        cls,
        logits: np.ndarray,
        mean: float = None,
        stdv: float = None,
        min_v: float = None,
        max_v: float = None) -> np.ndarray:
        '''
            Applies the MinMax normalization method and them place the
            resulting logits in the range of [0, 1] using MinMax.
        '''
        norm_logits = cls.z_score(logits, mean=mean, stdv=stdv)
        return cls.min_max(norm_logits, min_v=min_v, max_v=max_v)

    @classmethod
    def logit_norm_std(
        cls,
        logits: np.ndarray,
        t: float = 1.0,
        min_v: float = None,
        max_v: float = None) -> np.ndarray:
        '''
            Applies the LogitNorm method and them place the resulting logits
            in the range of [0, 1] using MinMax.
        '''
        norm_logits = cls.logit_norm(logits, t=t)
        return cls.min_max(norm_logits, min_v=min_v, max_v=max_v)

    @classmethod
    def cv_norm(cls, logits, to_prob: bool = False):
        with tempfile.NamedTemporaryFile(
            prefix='logits_', suffix='.jpg', mode='w'
        ) as tmp_dir:
            cv2.imwrite(tmp_dir.name, logits*255)
            norm_logits = cv2.imread(tmp_dir.name)[:, :, 0]

        if to_prob:
            return norm_logits / 255.0
        
        return abs(255 - norm_logits.astype('uint8'))
