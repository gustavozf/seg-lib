import os

BASET_PATH = 'E:\\UNIPD\\SAM\\'
EXPERIMENT_ID = 'v1_1_0'

class InferenceConfig:
    DATASET_PATH = os.path.join(BASET_PATH, 'data', 'test')
    BASE_OUTPUT_PATH = os.path.join(
        BASET_PATH, 'outputs', 'test', EXPERIMENT_ID
    )
    MODEL_CHECKPOINT_PATH = (
        os.path.join(BASET_PATH, 'pretrained_models', 'sam_vit_b_01ec64.pth'),
        os.path.join(
            BASET_PATH, 'outputs', 'train', EXPERIMENT_ID, 'best_SAMUS.pth'
        )
    )
    
    VERBOSITY_LEVEL = 'debug'
    OVERWRITE_OUTPUTS = True
    SAMPLING_MODE = 'grid'
    SAMPLING_STEP = 50
    BORDER_MODE = 'off'
    PRED_TH = 0.0
    BIN_TH = 0.5
    BATCH_SIZE = 8
    DEVICE = 'cpu' # inference should be always 'cpu'
    MODEL_TOPOLOGY = 'SAMUS' # one of {'SAM', 'SAM-Med2D', 'SAMUS'}
    MODEL_TYPE = 'default' # one of: {'default', 'vit_l', 'vit_b'}
    FUSION_RULE = 'default' # one of LogitsFusion.RULES
    DATASETS = [
        # {'name': 'ribs', 'source_mask': 'deeplab'},
        # {'name': 'ribs', 'source_mask': 'ensemble'},
        # {'name': 'ribs', 'source_mask': 'ensemble_small'},
        # {'name': 'ribs', 'source_mask': 'ensemble_v'},
        {'name': 'ribs', 'source_mask': 'cafe'},
        # {'name': 'ribs', 'source_mask': 'hsn'},
        # {'name': 'ribs', 'source_mask': 'polyp'},
        # {'name': 'ribs', 'source_mask': 'pooling'},
        # {'name': 'ribs', 'source_mask': 'ensemble_polyp_cafe'},
        # {'name': 'ribs', 'source_mask': 'ensemble_polyp_hsn'},
        # {'name': 'ribs', 'source_mask': 'ensemble_hsn_polyp_cafe'},
    ]

    @classmethod
    def to_json(cls):
        return {
            'DATASET_PATH': cls.DATASET_PATH,
            'BASE_OUTPUT_PATH': cls.BASE_OUTPUT_PATH,
            'MODEL_CHECKPOINT_PATH': cls.MODEL_CHECKPOINT_PATH,
            'VERBOSITY_LEVEL': cls.VERBOSITY_LEVEL,
            'OVERWRITE_OUTPUTS': cls.OVERWRITE_OUTPUTS,
            'SAMPLING_MODE': cls.SAMPLING_MODE,
            'SAMPLING_STEP': cls.SAMPLING_STEP,
            'BORDER_MODE': cls.BORDER_MODE,
            'PRED_TH': cls.PRED_TH,
            'FUSION_TH': cls.BIN_TH,
            'DEVICE': cls.DEVICE,
            'MODEL_TOPOLOGY': cls.MODEL_TOPOLOGY,
            'MODEL_TYPE': cls.MODEL_TYPE,
            'FUSION_RULE': cls.FUSION_RULE,
            'DATASETS': cls.DATASETS
        }