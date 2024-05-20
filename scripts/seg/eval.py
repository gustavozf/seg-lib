""" 
    Script used for evaluating Segmentation models (i.e., SegPVT and CAFE-net)
"""

import argparse
import os
import time

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from seg_lib.dataloaders.sam_dataset import SamGeneralDataset
from seg_lib.eval.metrics import Metrics
from seg_lib.io.files import dump_json, read_json
from seg_lib.models.pvt_v2 import SegPVT
from seg_lib.models.cafe_net.pvt import CAFE

TH = 0.5
# total number of parallel workers used in the dataloader
N_CPUS = os.cpu_count()
N_GPUS = torch.cuda.device_count()
DEVICE = 'cuda' if N_GPUS > 0 else 'cpu'
SUPPORTED_MODELS = {'CAFE', 'PTVv2'}

def get_args():
    parser = argparse.ArgumentParser(
        prog='SegmentationModelEval',
        description='Evaluate a segmentation model.'
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

def build_model(model_arch: str, ckpt_path: str):
    if model_arch not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported model architecture: {model_arch}')
    
    if model_arch == 'SegPVT':
        return SegPVT(backbone_ckpt_path=ckpt_path)
    
    return CAFE(pvtv2_path=ckpt_path)

def get_model(model_arch: str, ckpt_path: str, model_path: str):
    model = build_model(model_arch, ckpt_path)
    model.to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint)

    return model

def get_dataset(
        data_path: str,
        data_desc_path: str,
        batch_size: int = 8,
        embedding_size: int = 128,
        input_size: int = 352):
    csv_path = os.path.join(data_path, data_desc_path)
    df = pd.read_csv(csv_path)
    test_split = 'test' if 'test' in df['split'].unique() else 'val'
    del df

    test_dataset = SamGeneralDataset(
        data_path,
        split=test_split, 
        point_sampler=None,
        df_file_path=data_desc_path,
        img_size=input_size,
        embedding_size=embedding_size,
        prompt=None,
        read_img_as_grayscale=False)
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size * max(1, N_GPUS),
        shuffle=False,
        num_workers=N_CPUS,
        pin_memory=True)

def eval(test_dataset, model):
    metrics = Metrics()
    batch_sizes = []
    latency_p_batch = []
    pred_masks = []
    pred_logits = []
    file_names = []
    
    data_config = {'dtype': torch.float32, 'device': DEVICE}
    for batch in tqdm(test_dataset):
        imgs = batch['image'].to(**data_config)
        labels = batch['label'].to(**data_config)
        orig_sizes = np.array(
            list(zip(*batch['original_img_size']) ), dtype=int
        )

        with torch.no_grad():
            _start = time.time()
            preds = model(imgs)
            _end = time.time()

        latency_p_batch.append(_end - _start)
        batch_sizes.append(imgs.shape[0])

        if isinstance(preds, tuple):
            final_preds = preds[0]
            for i in range(1, len(preds)):
                final_preds = final_preds + preds[i]
            preds = final_preds
        
        logits = preds.sigmoid().detach().numpy()[:, 0, :, :]
        bin_masks = (logits > TH).astype('uint8')
        labels = labels.detach().numpy()[:, 0, :, :].astype('uint8')
        for i in range(bin_masks.shape[0]):
            bin_mask = cv2.resize(
                bin_masks[i], orig_sizes[i], cv2.INTER_NEAREST
            )
            label = cv2.resize(labels[i], orig_sizes[i], cv2.INTER_NEAREST)

            metrics.step(bin_mask, label)
            pred_logits.append(label)
            pred_masks.append(bin_mask)
            file_names.append(batch['img_name'][i])

    metrics = {
        **metrics.get_results(),
        'fps': sum(batch_sizes) / sum(latency_p_batch),
        'latency': sum(latency_p_batch) / sum(batch_sizes),
        'latency_p_batch': sum(latency_p_batch) / len(batch_sizes)
    }
    return metrics, pred_masks, pred_logits, file_names

def save_logits(
        bin_masks: list, pred_logits: list, file_names: list, output_path: str
    ):
    save_out_path = os.path.join(output_path, 'preds')
    os.makedirs(save_out_path, exist_ok=True)
    for i in tqdm(range(len(file_names))):
        img_name = file_names[i][:file_names[i].rindex('.')]
        cv2.imwrite(
            os.path.join(save_out_path, f'{img_name}.jpg'), bin_masks[i] * 255
        )
        np.save(
            os.path.join(save_out_path, f'{img_name}.npy'), pred_logits[i]
        )


def main():
    eval_config = get_args()
    train_config = read_json(
        os.path.join(eval_config.input_path, 'config.json')
    )
    
    output_path = os.path.join(eval_config.input_path, 'eval')
    os.makedirs(output_path, exist_ok=True)

    backbone_weights_path = (
        eval_config.backbone_weights_path
        if eval_config.backbone_weights_path is not None
        else train_config['checkpoint'])
    model_path = os.path.join(
        eval_config.input_path, f"best_{train_config['model_type']}.pth"
    )
    
    print('Getting model...')
    model = get_model(
        train_config['model_type'],
        backbone_weights_path,
        model_path)
    print('Getting dataset...')
    test_dataset = get_dataset(
        eval_config.data_path,
        eval_config.data_desc_path,
        batch_size=eval_config.batch_size,
        embedding_size=train_config['embedding_size'],
        input_size=train_config['input_size'])
    print('Running evaluation...')
    metrics, bin_masks, pred_logits, file_names = eval(test_dataset, model)
    print(metrics)
    print('Generating outputs...')
    dump_json(os.path.join(output_path, 'metrics.json'), metrics)
    save_logits(bin_masks, pred_logits, file_names, output_path)

if __name__ == '__main__':
    main()