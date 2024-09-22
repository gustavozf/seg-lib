import argparse
import os
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from seg_lib.eval.outputs import dump_stats, dump_metrics, print_metrics
from seg_lib.eval.metrics import Metrics
from seg_lib.eval.fusion import LogitsFusion
from seg_lib.io.image import read_bmask
from seg_lib.models.normalizer import Normalizer

BIN_TH = 0.5

logging.getLogger().setLevel(logging.INFO)

def save_outputs(
        metrics_path: str,
        filenames: list[str],
        baseline_metrics: Metrics,
        sam_metrics: Metrics,
        fusion_metrics: dict[str, Metrics]):
    logging.info('Results from the segmentation model')
    in_met_dict = print_metrics(baseline_metrics)
    dump_stats(
        os.path.join(metrics_path, 'baseline_stats.csv'),
        filenames,
        baseline_metrics
    )

    logging.info('Results from SAM')
    sam_met_dict = print_metrics(sam_metrics)
    dump_stats(
        os.path.join(metrics_path, 'sam_stats.csv'),
        filenames,
        sam_metrics
    )

    logging.info('Results from Fusion')
    fus_met_dict = {}
    for method in fusion_metrics:
        logging.info(f'\nFusion method: {method}')
        fus_met_dict[method] = print_metrics(fusion_metrics[method])
        dump_stats(
            os.path.join(metrics_path, f'{method}_fusion_stats.csv'),
            filenames, fusion_metrics[method]
        )

    dump_metrics(
        os.path.join(metrics_path, 'metrics.csv'),
        in_met_dict, sam_met_dict, fus_met_dict)
    del in_met_dict, sam_met_dict, fus_met_dict

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
        '-seg', '--seg_output_path',
        required=True, type=str,
        help='Path to the segmentation output predictions.')
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
    
    return parser.parse_args()

def main():
    eval_config = get_args()
    baseline_metrics = Metrics()
    sam_metrics = Metrics()
    fusion_metrics = {
        method : Metrics() for method in LogitsFusion.RULES
    }

    logging.info('Loading data...')
    test_df = get_data(
        eval_config.data_desc_path,
        split=eval_config.test_split_name)

    logging.info('Processing image inputs...')
    filenames = []
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        base_path = os.path.join(eval_config.data_path, row['subset'])
        img_base_name = row['img_name'][:row['img_name'].rindex('.')]
        filenames.append(row['img_name'])

        # read the label
        label_path = os.path.join(base_path, 'label', row['label_name'])
        label = read_bmask(label_path)
        del label_path

        # load the segmentation outputs and compare to the ground truth
        seg_base_path = os.path.join(
            eval_config.seg_output_path, 'eval', 'preds', img_base_name
        )
        seg_pred_bmask = read_bmask(seg_base_path + '.bmp')
        seg_pred_logits = np.load(seg_base_path + '.npy').astype(np.float32)
        baseline_metrics.step(seg_pred_bmask, label)
        del seg_base_path

        # load the SAM outputs and compare to the ground truth
        sam_base_path = os.path.join(
            eval_config.output_path, 'preds', img_base_name
        )
        sam_pred_bmask = read_bmask(sam_base_path + '.bmp')
        sam_pred_logits = Normalizer.sigmoid(
            np.load(sam_base_path + '.npy')
        ).astype(np.float32)
        sam_metrics.step(sam_pred_bmask, label)
        del sam_base_path

        # perform the fusion
        for method in fusion_metrics:
            fusion_mask = LogitsFusion.apply(
                seg_pred_logits, sam_pred_logits, method=method
            )
            fusion_metrics[method].step(fusion_mask > BIN_TH, label)
            del fusion_mask

        del label, seg_pred_bmask, seg_pred_logits
        del sam_pred_bmask, sam_pred_logits

    metrics_path = os.path.join(eval_config.output_path, 'metrics')
    os.makedirs(metrics_path, exist_ok=True)
    save_outputs(
        metrics_path, filenames,
        baseline_metrics, sam_metrics, fusion_metrics
    )

if __name__ == '__main__':
    main()