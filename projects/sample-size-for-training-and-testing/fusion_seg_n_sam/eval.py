import os
import logging

import numpy as np
from tqdm import tqdm
from seg_lib.eval.outputs import dump_stats, dump_metrics, print_metrics
from seg_lib.eval.metrics import select_metric_from_dataset
from seg_lib.eval.fusion import LogitsFusion
from seg_lib.io.image import read_bmask
from seg_lib.io.files import load_data_paths
from seg_lib.models.normalizer import Normalizer

from config import InferenceConfig as CONFIG

logging.getLogger().setLevel(logging.INFO)

for dataset in CONFIG.DATASETS:
    dataset, source_mask = dataset['name'], dataset['source_mask']
    metrics_class = select_metric_from_dataset(dataset.lower())
    baseline_metrics = metrics_class()
    sam_metrics = metrics_class()
    fusion_metrics = {
        method : metrics_class() for method in LogitsFusion.RULES
    }

    dataset_path = os.path.join(CONFIG.DATASET_PATH, dataset)
    paths = load_data_paths(dataset_path, source_mask)
    del dataset_path

    basenames = []
    out_path = os.path.join(CONFIG.BASE_OUTPUT_PATH, dataset)
    pred_path = os.path.join(
        out_path, CONFIG.MODEL_TOPOLOGY, source_mask, 'preds'
    )
    logging.info(f'Processing data in directory: {pred_path}')
    for idx, img_paths in tqdm(enumerate(paths), total=len(paths)):
        img_path, gt_mask_path, bmask_path, rmask_path = img_paths

        # get the input images and groundtruth
        gt_mask = read_bmask(gt_mask_path).astype(np.float32)
        b_mask = read_bmask(bmask_path).astype(np.float32)
        baseline_metrics.step(b_mask, gt_mask)
        del gt_mask_path, bmask_path, b_mask

        # get the prediction input
        basename = os.path.basename(img_path)
        basenames.append(basename)
        basename = basename[:basename.rfind('.') + 1]

        pred_mask_fname = basename + 'jpg'
        pred_mask_path = os.path.join(pred_path, pred_mask_fname)

        pred_mask = read_bmask(pred_mask_path).astype(np.float32)
        sam_metrics.step(pred_mask > CONFIG.PRED_TH, gt_mask)
        del img_path, pred_mask_fname, pred_mask_path, pred_mask

        # get the fusion mask
        logit_fname = 'low_' + basename + 'npy'
        logit_path = os.path.join(pred_path, logit_fname)
        del basename, logit_fname

        for method in fusion_metrics:
            fusion_mask = LogitsFusion.apply(
                # probablities from the source segmentator
                1.0 - read_bmask(rmask_path).astype(np.float32),
                # probablities from SAM
                Normalizer.sigmoid(
                    np.load(logit_path).astype(np.float32)
                ).astype(np.float32),
                method=method
            )
            fusion_metrics[method].step(fusion_mask > CONFIG.BIN_TH, gt_mask)
        del logit_path, gt_mask, fusion_mask

    logging.info(f'Results from {source_mask}')
    in_met_dict = print_metrics(baseline_metrics)
    metrics_path = os.path.join(
        out_path, CONFIG.MODEL_TOPOLOGY, source_mask, 'metrics'
    )
    os.makedirs(metrics_path, exist_ok=True)
    dump_stats(
        os.path.join(metrics_path, 'baseline_stats.csv'),
        basenames,
        baseline_metrics
    )
    del baseline_metrics

    logging.info('Results from SAM')
    sam_met_dict = print_metrics(sam_metrics)
    dump_stats(
        os.path.join(metrics_path, 'sam_stats.csv'), basenames, sam_metrics
    )
    del sam_metrics
    
    logging.info('Results from Fusion')
    fus_met_dict = {}
    for method in fusion_metrics:
        logging.info(f'\nFusion method: {method}')
        fus_met_dict[method] = print_metrics(fusion_metrics[method])
        dump_stats(
            os.path.join(metrics_path, f'{method}_fusion_stats.csv'),
            basenames, fusion_metrics[method]
        )
    del fusion_metrics

    dump_metrics(
        os.path.join(metrics_path, 'metrics.csv'),
        in_met_dict, sam_met_dict, fus_met_dict)
    del dataset, in_met_dict, sam_met_dict, fus_met_dict
