from seg_lib.models.pvt_v2.pvt_v2_seg import SegPVT

from seg_lib.models.sam import sam_model_registry, SamPredictor
from seg_lib.models.samus import samus_model_registry, SamusPredictor
from seg_lib.models.sam_med2d import (
    sam_med_2d_model_registry, SamMed2DPredictor
)
from seg_lib.models.selector import (
    sam_selector, predictor_selector, seg_selector,
    SUPPORTED_SEG_MODELS, SUPPORTED_SAM_MODELS, SUPPORTED_MODEL_TYPES
)
SAM_SIZES = {
    'SAM': {'input_size': 1024, 'embedding_size': 256},
    'SAM-Med2D': {'input_size': 1024, 'embedding_size': 256},
    'SAMUS': {'input_size': 256, 'embedding_size': 128}
}

# Segmenters
from seg_lib.models.pvt_v2.pvt_v2_seg import SegPVT
from seg_lib.models.cafe_net.pvt import CAFE