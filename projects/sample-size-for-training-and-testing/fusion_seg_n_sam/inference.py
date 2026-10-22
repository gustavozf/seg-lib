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
    SUPPORTED_SAM_MODELS, SUPPORTED_SAMV2_TYPES
)

N_GPUS = torch.cuda.device_count()
DEVICE = 'cuda' if N_GPUS > 0 else 'cpu'
# threshold applied to the segmentation probabilities before sampling
SEG_TH = 0.5
EMPTY_LOGIT = -20.0

# ================================================================== PREDICTION
def sample_and_predict(
        predictor: SamPredictor,
        sampler: Sampler,
        img: np.ndarray,
        prompt_mask: np.ndarray,
        model_topology: str):
    '''Samples point prompts from the binary `prompt_mask` and runs SAM.
       Returns the binary mask and the logits of the best SAM output.'''
    mask_of_blobs = measure.label(prompt_mask)
    # no foreground to sample from: empty mask with logits whose sigmoid ~ 0
    if not np.any(mask_of_blobs):
        return (
            np.zeros(prompt_mask.shape, dtype=bool),
            np.full(prompt_mask.shape, EMPTY_LOGIT, dtype=np.float32)
        )

    # Sample the checkpoints (at least one for blob)
    input_point, input_label = sampler.sample(mask_of_blobs, prompt_mask)

    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # SAMUS always takes this parameters as False
        multimask_output=model_topology != "SAMUS",
        return_logits=True
    )

    best_score_idx = np.argmax(scores)
    binary_mask = masks[best_score_idx] > predictor.mask_threshold
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
        choices=SUPPORTED_SAM_MODELS,
        help='Topology name of the SAM model to be loaded.')
    parser.add_argument(
        '--sam_model_type',
        required=False, type=str,
        default='vit_b',
        choices=SUPPORTED_MODEL_TYPES.union(SUPPORTED_SAMV2_TYPES),
        help=(
            'Type of SAM model encoder to be loaded '
            '(ViT types for SAM-based models, Hiera types for SAMv2/2.1).'
        ))
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
        '-s', '--test_split_name',
        required=False, type=str,
        default='val', choices={'val', 'test'},
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

    args = parser.parse_args()
    if not args.is_oracle and args.seg_output_path is None:
        parser.error('--seg_output_path is required without --is_oracle.')

    return args

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
        split=inference_config.test_split_name)
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
        label[label > 0] = 1
        
        # mask used for sampling the input prompts: the label in oracle mode,
        # the binarized segmentation probabilities otherwise.
        prompt_mask = label
        if not inference_config.is_oracle:
            probs_path = os.path.join(
                inference_config.seg_output_path,
                'eval', 'preds',
                img_base_name + '.npy'
            )
            prompt_mask = (np.load(probs_path) > SEG_TH).astype(np.uint8)
    
        pred_bin_mask, pred_logits = sample_and_predict(
            predictor, sampler,
            image, prompt_mask,
            inference_config.sam_model_topology)

        # save the outputs
        cv2.imwrite(
            os.path.join(out_path, img_base_name + '.bmp'),
            pred_bin_mask.astype(np.uint8) * 255)
        np.save(
            os.path.join(out_path, img_base_name + '.npy'),
            pred_logits)

if __name__ == '__main__':
    main()