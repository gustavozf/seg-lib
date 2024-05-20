import logging

import pandas as pd

from seg_lib.eval.metrics import Metrics

def print_metrics(metrics_obj: Metrics):
    avg_metrics  = metrics_obj.get_results()
    logging.info(f'Total number of objects: {len(metrics_obj.ious)}')
    
    for metric in avg_metrics:
        logging.info(
            f"average {metric}: {avg_metrics[metric]*100:5.2f}"
        )

    return avg_metrics

def dump_metrics(
        output_path: str, bl_met: dict, pred_met: dict, fus_met: dict):
    indexes = ['iou', 'dice', 'mae', 'f-measure', 'e-measure']
    pd.DataFrame({
        'baseline': [bl_met[idx] for idx in indexes],
        'sam_pred': [pred_met[idx] for idx in indexes],
        **{
            f'fusion_{method}': [fus_met[method][idx] for idx in indexes]
            for method in sorted(fus_met.keys())
        }
    }, index=indexes).to_csv(output_path)

def dump_stats(out_path: str, file_names: list, metrics: Metrics):
    pd.DataFrame({
        'filenames': file_names,
        'iou': metrics.ious,
        'dice': metrics.dices,
        'mae': metrics.maes,
        'f-measure': metrics.wfms,
        'e-measure': metrics.emes
    }).to_csv(out_path)
