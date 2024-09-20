import argparse
import os

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import measure

from seg_lib.dataloaders.image_ops import to_grayscale
from seg_lib.prompt import Sampler
from seg_lib.io.image import read_img
from seg_lib.models.sam import SamPredictor
from seg_lib.models.device import CLEAN_CACHE
from seg_lib.models.selector import (
    predictor_selector, SUPPORTED_MODEL_TYPES,
    SUPPORTED_SAM_MODELS, SUPPORTED_SAMv2_TYPES
)

N_GPUS = torch.cuda.device_count()
DEVICE = 'cuda' if N_GPUS > 0 else 'cpu'

# ================================================================== PREDICTION
def sample_and_predict(
        predictor: SamPredictor,
        sampler: Sampler,
        img: np.ndarray,
        logits: np.ndarray,
        model_topology: str):
    best_score_idx=0
    binary_mask = logits
    input_point = np.array([])
    input_point = np.array([])
    masks=[binary_mask]
    predict_threshold = predictor.model.mask_threshold

    mask_of_blobs = measure.label(logits)
    # Sample the checkpoints (at least one for blob)
    unique_blobs = np.unique(mask_of_blobs)
    num_blobs = unique_blobs.shape[0]
    if not (num_blobs==1 and 0 in unique_blobs):
        input_point, input_label = sampler.sample(mask_of_blobs, logits)

        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            # SAMUS always takes this parameters as False
            multimask_output=model_topology != "SAMUS",
            return_logits=True
        )

        best_score_idx = np.argmax(scores)
        binary_mask = masks[best_score_idx] > predict_threshold
        CLEAN_CACHE[DEVICE]()

    return binary_mask, masks[best_score_idx]

def get_data(df_path: str, split: str = 'test'):
    df = pd.read_csv(df_path)
    df = df[df['split'] == split]
    
    return df
# ======================================================================== MAIN
def get_args():
    parser = argparse.ArgumentParser(
        prog='SamAndSegModelInference',
        description='Perform the inference on a segmentation model.'
    )

    parser.add_argument(
        '-o', '--output_path',
        required=True, type=str,
        help='Path to the outputs.')
    parser.add_argument(
        '--sam_model_path',
        required=True, type=str,
        help='Path to the training output path.')
    parser.add_argument(
        '--sam_model_topology',
        required=False, type=str,
        default='SAMUS',
        choices=(SUPPORTED_SAM_MODELS | SUPPORTED_SAMv2_TYPES),
        help='Topology name of the SAM model to be loaded.')
    parser.add_argument(
        '--sam_model_type',
        required=False, type=str,
        default='vit_b', choices=SUPPORTED_MODEL_TYPES,
        help='Type of SAM model encoder to be loaded.')
    parser.add_argument(
        '-seg', '--seg_output_path',
        required=False, type=str,
        help=(
            'Path to the segmentation output predictions. '
            'Only required if oracle mode (--is_oracle) is not used.'
        )
    )
    parser.add_argument(
        '-d', '--data_path',
        required=True, type=str,
        help='Path to the test data.')
    parser.add_argument(
        '-dd', '--data_desc_path',
        required=True, type=str,
        help=('Path to the data description file (CSV format). '
              'Read from `data_path`.'))
    parser.add_argument(
        '-s', '--split_name',
        required=False, type=str, default='val',
        help=(
            'Data split name, used to filter the data samples on the '
            'metadata file.'
        )
    )
    parser.add_argument(
        '--sampling_step',
        required=False, type=int, default=50,
        help='Number of steps between the point prompts to be sampled.')
    parser.add_argument(
        '--sampling_mode',
        required=False, type=str,
        default='grid', choices=Sampler.SAMPLING_MODES,
        help='Sampling mode.')
    parser.add_argument(
        '--erode_grid',
        required=False, action='store_true',
        help=(
            'If passed will perform an erosion when sampling the prompts. '
            'Requires the "grid" sampling mode.'
        )
    )
    parser.add_argument(
        '--is_oracle',
        required=False, action='store_true',
        help='If passed, prompt points will be sampled from the image label.')
    
    
    return parser.parse_args()

def main():
    inference_config = get_args()

    out_path = os.path.join(inference_config.output_path, 'preds')
    os.makedirs(out_path, exist_ok=True)

    print('Loading predictor...')
    predictor = predictor_selector(
        model_topology=inference_config.sam_model_topology,
        checkpoint_path=inference_config.sam_model_path,
        model_type=inference_config.sam_model_type,
        device=DEVICE)
    print('Loading data...')
    test_df = get_data(
        inference_config.data_desc_path,
        split=inference_config.split_name)
    print('Loading sampler...')
    sampler = Sampler(
        sampling_step=inference_config.sampling_step,
        min_blob_count=10,
        mode=inference_config.sampling_mode,
        erode_grid=inference_config.erode_grid)

    print('Processing image inputs...')
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        base_path = os.path.join(inference_config.data_path, row['subset'])
        img_base_name = row['img_name'][:row['img_name'].rindex('.')]

        # read the input image
        img_path = os.path.join(base_path, 'img', row['img_name'])
        image = read_img(img_path)
        # SAMUS expects a 3-channel grayscale image. Convert it if required
        if inference_config.sam_model_topology == 'SAMUS':
            image = to_grayscale(image)
        
        # read the label
        label_path = os.path.join(base_path, 'label', row['label_name'])
        label = cv2.imread(label_path, 0)
        label[label > 1] = 1
        
        # read the logits for sampling the input prompts.
        # if the we are running in oracle mode, the label will be the logits.
        logits = label
        if inference_config.is_oracle:
            logits_path = os.path.join(
                inference_config.seg_output_path,
                'eval', 'preds',
                img_base_name + '.npy'
            )
            logits = np.load(logits_path)
    
        pred_bin_mask, pred_logits = sample_and_predict(
            predictor, sampler,
            image, logits,
            inference_config.sam_model_topology)

        # save the outputs
        cv2.imwrite(
            os.path.join(out_path, img_base_name + '.bmp'),
            pred_bin_mask * 255)
        np.save(
            os.path.join(out_path, img_base_name + '.npy'),
            pred_logits)

if __name__ == '__main__':
    main()