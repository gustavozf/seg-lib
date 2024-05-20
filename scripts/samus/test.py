import argparse
import os
from typing import Callable

import cv2
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from seg_lib.io.files import read_json
from seg_lib.eval.metrics import Metrics
from seg_lib.models.samus import samus_model_registry, SamusPredictor
from seg_lib.prompt.train_sampler import TrainPromptSampler

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
    
    return parser.parse_args()

class MaskInferenceAfterSample:
    def __init__(self,
            predictor,
            point_sampler: Callable,
            bbox_sampler: Callable,
            mask_sampler: Callable):
        self.predictor = predictor
        self.point_sampler = self.__set_point_sampler(point_sampler)
        self.bbox_sampler = self.__set_sampler(bbox_sampler)
        self.mask_sampler = self.__set_sampler(mask_sampler)
        self.th = predictor.model.mask_threshold

    def __set_sampler(self, sampler):
        if sampler is None:
            return lambda _: None

        return sampler
    
    def __set_point_sampler(self, sampler):
        if sampler is None:
            return lambda _: None, None

        return sampler

    def __call__(self, img, label):
        point_coords, point_labels = self.point_sampler(label)
        box = self.bbox_sampler(label)
        mask_input = self.mask_sampler(label)

        self.predictor.set_image(img)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=True,
            return_logits=False
        )
        best_score_idx = np.argmax(scores)
        binary_mask = masks[best_score_idx] > self.th

        return binary_mask

def get_predictor(
        checkpoint_path: str,
        backbone_path: str,
        input_size: int = 256):
    model = samus_model_registry['vit_b'](
        encoder_input_size=input_size, checkpoint=backbone_path
    )
    model.to(torch.device(DEVICE))
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(DEVICE)
    )
    model.load_state_dict(checkpoint)
    return SamusPredictor(model)

def get_data(df_path: str, split: str = 'test'):
    df = pd.read_csv(df_path, index_col=0)
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
        train_config['output_path'], f'best_{train_config['model_name']}.pth'
    )
    predictor = get_predictor(
        checkpoint_path,
        eval_config.backbone_weights_path,
        input_size=train_config['input_size']
    )
    test_df = get_data(
        os.path.join(eval_config.data_path, eval_config.df_file_path),
        split='test')

    output_metrics = {}
    sampler_args = {
        'points': (predictor, point_sampler, None, None),
    }
    for method, args in sampler_args.items():
        print('Generating metrics with method: ', 'points')
        metrics.reset()
        get_inference = MaskInferenceAfterSample(*args)

        for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
            base_path = os.path.join(train_config.data_path, row['subset']) 
            img_path = os.path.join(base_path, 'img', row['img_name'])
            label_path = os.path.join(base_path, 'label', row['label_name'])

            img = cv2.imread(img_path, 1)[:, :, ::-1]
            label = cv2.imread(label_path, 0)
            label[label > 1] = 1

            binary_mask = get_inference(img, label)
            metrics.step(binary_mask, label)
        
        output_metrics[method] = metrics.get_results()

    metrics_path = os.path.join(train_config.output_path, 'metrics.csv')
    out_df = pd.DataFrame(output_metrics)
    out_df.to_csv(metrics_path)
    print(out_df)

if __name__ == '__main__':
    main()