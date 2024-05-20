import os
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm
from skimage import measure

from seg_lib.prompt import Sampler
from seg_lib.io.image import read_img, read_bmask, read_rmask
from seg_lib.io.files import load_data_paths
from seg_lib.models.sam import SamPredictor
from seg_lib.models.selector import predictor_selector

from config.inference import InferenceConfig as CONFIG

logging.getLogger().setLevel(logging.DEBUG)
# =============================================================== IO & PATH OPS
def read_imgs_from_paths(paths: tuple) -> tuple:
    img_path, gt_mask_path, src_bmask_path, src_rmask_path = paths
    logging.debug(f"img_path          : {img_path}")
    logging.debug(f"gt_mask_path      : {gt_mask_path}")
    logging.debug(f"dplabv3_bmask_path: {src_bmask_path}")
    logging.debug(f"dplabv3_rmask_path: {src_rmask_path}")
    
    # Load images from disk using paths
    return (
        read_img(img_path),
        read_bmask(gt_mask_path),
        read_bmask(src_bmask_path),
        read_rmask(src_rmask_path)
    )

def img_shape_matches(loaded_images: tuple, source_mask: str = 'oracle'):
    img, gt_mask, src_bmask, src_rmask = loaded_images
    img_matches_gt = img.shape[:2] == gt_mask.shape

    if source_mask=="oracle":
        return img_matches_gt
            
    return img_matches_gt and (src_bmask.shape == src_rmask.shape[:2])

def get_complete_output_path(bop, dataset_name, src_msk, model, create=False):
    results_dir = os.path.join(bop, dataset_name, src_msk, model)
    if create:
        os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_sampled_points_folder_name(
    s_step, pnts_smp_mode: str='', border_mode: str='on'):
    return (
        pnts_smp_mode 
            + (("_" + str(s_step)) if pnts_smp_mode=="grid" else "")
            + ("_bm" if border_mode=="on" and pnts_smp_mode=="grid" else "")
    )

# ================================================================== PREDICTION
def predict_sam_masks(
        predictor: SamPredictor,
        sampler: Sampler,
        data_paths: list,
        masks_out_dir: str,
        sampled_points_out_dir: str,
        source_mask: str = 'deeplab',
        predict_threshold: float = 0.0):
    is_model_cuda = predictor.model.device == 'cuda'
    data_len = len(data_paths)
    logging.debug(f"Total number of files: {data_len}")

    for idx, paths in tqdm(enumerate(data_paths), total=data_len):
        logging.debug(f" - img idx {str(idx+1).zfill(6)}/{data_len}:")
        loaded_images = read_imgs_from_paths(paths)

        if not img_shape_matches(loaded_images, source_mask=source_mask):
            logging.error('A mismatch between image shapes has been found!')
            continue

        # Output paths to the resulting binary mask and logits
        basename = os.path.basename(paths[0])
        basename = basename[:basename.rfind(".") + 1]
        out_logits_path = os.path.join(masks_out_dir, 'low_'+ basename + "npy")
        out_mask_best_path = os.path.join(masks_out_dir, basename + "jpg")
        if (not CONFIG.OVERWRITE_OUTPUTS
                and os.path.exists(out_mask_best_path)
                and os.path.exists(out_logits_path)):
            logging.warning(f'{basename} already exists, skipping file.')
            continue

        img, gt_mask, src_bmask, _ = loaded_images
        mask_to_sample = gt_mask if source_mask == "oracle" else src_bmask
        # Count the number of distinct labels
        #  -> it corresponds to the # of blobs
        mask_of_blobs = measure.label(mask_to_sample)
        
        # Sample the checkpoints (at least one for blob)
        unique_blobs = np.unique(mask_of_blobs)
        num_blobs = unique_blobs.shape[0]
        bin_path = os.path.join(
            sampled_points_out_dir, os.path.basename(paths[1])[:-4] + ".bin_"
        )
        if num_blobs==1 and 0 in unique_blobs:
            input_point, input_label = np.array([]), np.array([])
            input_point.tofile(bin_path)
            binary_mask = mask_to_sample
            masks=[binary_mask]
            best_score_idx=0
        else:
            input_point, input_label = sampler.sample(
                mask_of_blobs, mask_to_sample
            )
            input_point.tofile(bin_path)
            predictor.set_image(img)
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                # SAMUS always takes this parameters as False
                multimask_output=CONFIG.MODEL_TOPOLOGY != "SAMUS",
                return_logits=True
            )

            if is_model_cuda:
                torch.cuda.empty_cache()

            best_score_idx = np.argmax(scores)
            binary_mask = masks[best_score_idx] > predict_threshold
        
        # save the outputs
        cv2.imwrite(out_mask_best_path, binary_mask * 255) # output bin mask
        np.save(out_logits_path, masks[best_score_idx]) # logits

# ======================================================================== MAIN
def main():
    logging.info("Creating SAM model and moving it to device")
    predictor = predictor_selector(
        model_topology=CONFIG.MODEL_TOPOLOGY,
        checkpoint_path=CONFIG.MODEL_CHECKPOINT_PATH,
        model_type=CONFIG.MODEL_TYPE,
        device=CONFIG.DEVICE
    )
    for dataset in CONFIG.DATASETS:
        dataset_name, source_mask = dataset['name'], dataset['source_mask']
        dataset_path = os.path.join(CONFIG.DATASET_PATH, dataset_name)
        data_paths = load_data_paths(dataset_path, source_mask)
        
        # The sampler is used for automatically generating the prompts
        sampler = Sampler(
            sampling_step=CONFIG.SAMPLING_STEP,
            min_blob_count=20 if dataset_name=="portrait" else 10,
            mode=CONFIG.SAMPLING_MODE,
            erode_grid=CONFIG.BORDER_MODE)

        # Create the output paths
        masks_out_dir = os.path.join(
            CONFIG.BASE_OUTPUT_PATH,
            dataset_name,
            CONFIG.MODEL_TOPOLOGY,
            source_mask,
            'preds')
        sampled_pnts_out_dir = os.path.join(
            CONFIG.BASE_OUTPUT_PATH,
            dataset_name,
            CONFIG.MODEL_TOPOLOGY,
            source_mask,
            'sampled_points_final',
            CONFIG.MODEL_TYPE,
            get_sampled_points_folder_name(
                CONFIG.SAMPLING_STEP,
                pnts_smp_mode=CONFIG.SAMPLING_MODE,
                border_mode=CONFIG.BORDER_MODE))
        logging.info("Creating output folders")
        os.makedirs(masks_out_dir, exist_ok=True)
        os.makedirs(sampled_pnts_out_dir, exist_ok=True)
        logging.info(
            f"Results will be saved at for results_dir {masks_out_dir}"
        )
        
        logging.info('Generating masks with SAM')
        with torch.no_grad():
            predict_sam_masks(
                predictor, sampler, data_paths,
                masks_out_dir, sampled_pnts_out_dir,
                source_mask=source_mask,
                predict_threshold=CONFIG.PRED_TH)

if __name__ == '__main__':
    main()