import numpy as np

class LogitsFusion:
    # x == probabilities from the segmentation model
    # y == probabilities from the SAM model
    RULES = {
        'default': lambda x, y: (2*x + y) / 3,
        'custom': lambda x, y : (2*x + np.min([x, y], axis=0)) / 3,
        'min': lambda x, y: np.min([x, y], axis=0),
        'max': lambda x, y: np.max([x, y], axis=0),
        'mean': lambda x, y: np.mean([x, y], axis=0),
        'median': lambda x, y: np.median([x, y], axis=0)
    }

    @classmethod
    def apply(cls, source_mask, pred_mask, method: str = 'default'):
        return cls.select_method(method=method)(source_mask, pred_mask)

    @classmethod
    def select_method(cls, method: str = 'default'):
        if method == None:
            return cls.RULES['default']

        if not method in cls.RULES:
            raise ValueError(
                f'"{method}" not supported!'
                f' Available methods: {cls.RULES}')

        return cls.RULES[method]
    
class KittlerLogitsFusion(LogitsFusion):
    RULES = {
        'default': lambda x, y: (2*x + y) / 3,
        'custom': lambda x, y : (2*x + np.min([x, y], axis=0)) / 3,
        'min': lambda x, y: np.min([x, y], axis=0),
        'max': lambda x, y: np.max([x, y], axis=0),
        'mean': lambda x, y: np.mean([x, y], axis=0),
        'median': lambda x, y: np.median([x, y], axis=0),
        'sum': lambda x, y: np.sum([x, y], axis=0),
        'sub': lambda x, y: x - y,
        'prod': lambda x, y: np.prod([x, y], axis=0)
    }

    @classmethod
    def apply(cls, source_mask, pred_mask, method: str = 'default'):
        comb_mask_probs = cls.select_method(method=method)(
            cls.to_probs(source_mask), cls.to_probs(pred_mask)
        )
        final_prediction = np.argmax(comb_mask_probs, axis=-1)

        return final_prediction.astype('float32')

    @classmethod
    def to_probs(cls, probs):
        # Converts a 1D array of probs to a 2D array. Required in order to 
        # apply the Kittler rules correctly. Example:
        #   input = [0.1, 0.6, 0.3]
        #   output = [[0.9, 0.1], [0.4, 0.6], [0.7, 0.3]]
        probs_2d = np.zeros((*probs.shape, 2), dtype='float32')
        probs_2d[..., 1] = probs
        probs_2d[..., 0] = 1 - probs
        return probs_2d
