import os
import gc

import pandas as pd
from ultralytics import YOLO
from seg_lib.models.selector import predictor_selector
from seg_lib.models.device import CLEAN_CACHE, get_device
import argparse

from src.fusion_protocols import (
    FusionProtocol1,
    FusionProtocol2,
    FusionProtocol3,
    FusionProtocol4
)

SUBJECT_TAGS = ['2C', '4C', '5C', '22TW', '23TW', '28TW']

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fusion protocols.')
    parser.add_argument(
        '--data_path',
        type=str, required=True,
        help='Path to the dataset (labelme format).')
    parser.add_argument(
        '--output_path',
        type=str, default='fusion/',
        help='Path to save the output.')
    parser.add_argument(
        '--yolo_model_path',
        type=str, required=True,
        help='Path to the YOLO model.')
    parser.add_argument(
        '--sam_model_path',
        type=str, required=True,
        help='Path to the SAM model.')
    return parser.parse_args()

def evaluate(
        protocol_name: str,
        protocol_class: FusionProtocol1,
        data_path: str = 'data/',
        output_path: str = 'output/',
        yolo_model_path: str = 'yolo/',
        sam_model_path: str = 'sam/sam_vit_b_01ec64.pth',
        device: str = 'cpu'):
    output_path = os.path.join(output_path, f'{protocol_name}.csv')
    out_metrics = {
        'animal_tag': [],
        'arch': [],
        'num_gt_cells': [],
        'num_pred_objs': [],
        'num_matches': [],
        'mIoU': [],
        'mDICE': [],
        'MAE': [],
        'F-Measure': [],
        'E-Measure': []
    }
    for tag in SUBJECT_TAGS:
        print('Processing: ', tag)
        target_input = os.path.join(data_path, tag)
        labels_paths = [
            _fpath.path for _fpath in os.scandir(target_input)
            if _fpath.name.endswith('.json')
        ]

        yolo_path = os.path.join(
            yolo_model_path, tag, 'train', 'weights', 'best.pt'
        )
        protocol = protocol_class(
            YOLO(yolo_path),
            predictor_selector(
                sam_model_path,
                model_topology='SAM',
                model_type='vit_b',
                device=device)
        )
        metrics = protocol(labels_paths)

        out_metrics['animal_tag'].append(tag)
        out_metrics['arch'].append('SAM_b_YOLOv8m')
        out_metrics['num_gt_cells'].append(metrics['num_gt_cells'])
        out_metrics['num_pred_objs'].append(metrics['num_pred_objs'])
        out_metrics['num_matches'].append(metrics['num_matches'])
        out_metrics['mIoU'].append(metrics['iou'])
        out_metrics['mDICE'].append(metrics['dice'])
        out_metrics['MAE'].append(metrics['mae'])
        out_metrics['F-Measure'].append(metrics['f-measure'])
        out_metrics['E-Measure'].append(metrics['e-measure'])

        pd.DataFrame(out_metrics).to_csv(output_path)
        gc.collect()
    return pd.DataFrame(out_metrics)

def main():
    args = parse_args()
    device = get_device()
    eval_configs = [
        ('protocol_1', FusionProtocol1),
        ('protocol_2', FusionProtocol2),
        ('protocol_3', FusionProtocol3),
        ('protocol_4', FusionProtocol4),
    ]
    os.makedirs(args.output_path, exist_ok=True)

    for config in eval_configs:
        print('Evaluating: ', config[0])
        evaluate(
            *config,
            data_path=args.data_path,
            output_path=args.output_path,
            yolo_model_path=args.yolo_model_path,
            sam_model_path=args.sam_model_path,
            device=device
        )
        gc.collect()
        CLEAN_CACHE[device]()
        print()

if __name__ == '__main__':
    main()