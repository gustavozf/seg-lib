import argparse
import os

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from seg_lib.dataloaders.image_ops import to_grayscale
from seg_lib.eval.metrics import Metrics
from seg_lib.io.image import read_img
from seg_lib.io.files import read_json
from seg_lib.models.selector import build_samus_model, SamusPredictor
from seg_lib.prompt import TrainPromptSampler, MaskInferenceAfterSample

# total number of parallel workers used in the dataloader
N_CPUS = os.cpu_count()
N_GPUS = torch.cuda.device_count()
DEVICE = 'cuda' if N_GPUS > 0 else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(
        prog='SamusModelTest',
        description='Evaluate a SAMUS model using the predictor class.'
    )

    parser.add_argument(
        '-i', '--input_path',
        required=True, type=str,
        help='Path to the training output path.')
    parser.add_argument(
        '-w', '--backbone_weights_path',
        required=False, type=str,
        help='Path to the backbone weights.')
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
        '-b', '--batch_size',
        required=False, type=int, default=8,
        help='Batch size used for training and validation.')
    parser.add_argument(
        '-s', '--test_split_name',
        required=False, type=str,
        default='val', choices={'val', 'test'},
        help=(
            'Data split name, used to filter the data samples on the '
            'metadata file.'
        )
    )
    
    return parser.parse_args()

def get_predictor(
        checkpoint_path: str,
        backbone_path: str,
        input_size: int = 256):
    model = build_samus_model(checkpoint_path, device=DEVICE)
    return SamusPredictor(model)

def get_data(df_path: str, split: str = 'test'):
    df = pd.read_csv(df_path)
    df = df[df['split'] == split]
    
    return df

def main():
    eval_config = get_args()
    train_config = read_json(
        os.path.join(eval_config.input_path, 'config.json')
    )

    metrics = Metrics()
    point_sampler = TrainPromptSampler(
        min_blob_count=10, max_num_prompts=train_config['n_clicks']
    )

    checkpoint_path = os.path.join(
        train_config['output_path'], f"best_{train_config['model_type']}.pth"
    )
    predictor = get_predictor(
        checkpoint_path,
        eval_config.backbone_weights_path,
        input_size=train_config['input_size']
    )
    test_df = get_data(
        eval_config.data_desc_path,
        split=eval_config.test_split_name
    )

    output_metrics = {}
    sampler_args = {'points': (predictor, point_sampler, None, None)}
    for method, args in sampler_args.items():
        print(f'Generating metrics with method: {method}')
        metrics.reset()
        get_inference = MaskInferenceAfterSample(*args)

        for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
            base_path = os.path.join(train_config['data_path'], row['subset']) 
            img_path = os.path.join(base_path, 'img', row['img_name'])
            label_path = os.path.join(base_path, 'label', row['label_name'])

            # read image and convert it to a 3-channel grayscale image
            image = read_img(img_path)
            image = to_grayscale(image)

            label = cv2.imread(label_path, 0)
            label[label > 1] = 1

            binary_mask = get_inference(image, label)
            metrics.step(binary_mask, label)
        
        output_metrics[method] = metrics.get_results()

        metrics_path = os.path.join(
            train_config['output_path'], f'{method}_metrics.csv'
        )
        out_df = pd.DataFrame(output_metrics)
        out_df.to_csv(metrics_path)
        print(out_df)

if __name__ == '__main__':
    main()