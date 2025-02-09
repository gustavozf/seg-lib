import argparse
import gc
import os

import pandas as pd
from tqdm import tqdm
from seg_lib.io.files import read_json
from seg_lib.models.selector import predictor_selector
from seg_lib.models.device import CLEAN_CACHE, get_device

from src.sam_inference import SamSingleInference, SamSingleRandomInference

SUBJECT_TAGS = ['2C', '4C', '5C', '22TW', '23TW', '28TW']
INFER_CLASSES = [SamSingleRandomInference, SamSingleInference]

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM models")
    parser.add_argument(
        '--data_path',
        type=str, required=True,
        help='Path to the dataset (labelme format).')
    parser.add_argument(
        '--models_path',
        type=str, default='sam/',
        help='Path to the pretrained SAM models')
    parser.add_argument(
        '--output_path',
        type=str, default='output/',
        help='Path to save the outputs.')
    return parser.parse_args()

def get_models_paths(models_path: str):
    return {
        'sam_vit_b': os.path.join(models_path, 'sam_vit_b_01ec64.pth'),
        'sam_vit_l': os.path.join(models_path, 'sam_vit_l_0b3195.pth'),
        'sam_vit_h': os.path.join(models_path, 'sam_vit_h_4b8939.pth'),
        'sam2_tiny': os.path.join(models_path, 'sam2_hiera_tiny.pt'),
        'sam2_small': os.path.join(models_path, 'sam2_hiera_small.pt'),
        'sam2_base_plus': os.path.join(models_path, 'sam2_hiera_base_plus.pt'),
        'sam2_large': os.path.join(models_path, 'sam2_hiera_large.pt'),
        'sam_med2d': os.path.join(models_path, 'sam-med2d_b.pth')
    }

def evaluate(
        predictor,
        model_type: str = 'SAM',
        encoder: str = 'vit_b',
        data_path: str = 'data/'
    ):
    out_dict = {
        'animal_tag': [],
        'arch': [],
        'encoder': [],
        'prompt': [],
        'input_img': [],
        'eval_mode': [],
        'mIoU': [],
        'mDICE': [],
        'MAE': [],
        'F-Measure': [],
        'E-Measure': []
    }
    print('Total number of runs: ', len(SUBJECT_TAGS)*len(INFER_CLASSES))
    for tag in SUBJECT_TAGS:
        target_input = os.path.join(data_path, tag)
        labels_paths = [
            _fpath.path for _fpath in os.scandir(target_input)
            if _fpath.name.endswith('.json')
        ]

        for infer_class in INFER_CLASSES:
            infer = infer_class(predictor)
            desc = f'Evaluating {tag} ({infer.eval_mode} mode)'
            for labels_path in tqdm(labels_paths, desc=desc):
                data = read_json(labels_path)
                infer(data)

            results = infer.get_results()
            for _prompt in results:
                out_dict['animal_tag'].append(tag)
                out_dict['arch'].append(model_type)
                out_dict['encoder'].append(encoder)
                out_dict['prompt'].append(_prompt)
                out_dict['input_img'].append(infer.input_color)
                out_dict['eval_mode'].append(infer.eval_mode)
                out_dict['mIoU'].append(results[_prompt]['iou'])
                out_dict['mDICE'].append(results[_prompt]['dice'])
                out_dict['MAE'].append(results[_prompt]['mae'])
                out_dict['F-Measure'].append(results[_prompt]['f-measure'])
                out_dict['E-Measure'].append(results[_prompt]['e-measure'])

    return pd.DataFrame(out_dict)

def main():
    args = get_args()
    device = get_device()
    model_paths = get_models_paths(args.models_path)
    model_configs = [
        ('SAM', model_paths['sam_vit_b'], 'vit_b'),
        ('SAM', model_paths['sam_vit_l'], 'vit_l'),
        ('SAM', model_paths['sam_vit_h'], 'vit_h'),
        ('SAMv2', model_paths['sam2_tiny'], 'hiera_t'),
        ('SAMv2', model_paths['sam2_small'], 'hiera_s'),
        ('SAMv2', model_paths['sam2_base_plus'], 'hiera_b+'),
        ('SAMv2', model_paths['sam2_large'], 'hiera_l'),
        ('SAM-Med2D', model_paths['sam_med2d'], 'vit_b')
    ]

    os.makedirs(args.output_path, exist_ok=True)
    for model_type, model_path, encoder in model_configs:
        print('Evaluating model: ', model_type, encoder)
        sam_pred = predictor_selector(
            model_path,
            model_topology=model_type,
            model_type=encoder,
            device=device)
        sam_results = evaluate(
            sam_pred,
            model_type=model_type,
            encoder=encoder,
            data_path=args.data_path)
        
        pd.DataFrame(sam_results).to_csv(
            os.path.join(args.output_path, f'{model_type}_{encoder}.csv')
        )
        del sam_pred, sam_results
        
        gc.collect()
        CLEAN_CACHE[device]()
        print()

if __name__ == '__main__':
    main()