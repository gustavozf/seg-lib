#!/bin/bash

EXP_TAG=samus_seg_fusion
SEG_EXP_TAG=seg_model
SAM_EXP_TAG=samus_model

# BASE_PATH=/path/to/input/files
BASE_PATH="/media/zanoni/TOSHIBA EXT/UNIPD/SAM/samus-train-n-fusion"
SAM_MODEL_PATH=outputs/$SAM_EXP_TAG/best_SAMUS.pth
DATASET_DESCRIPTOR=ribs_da1.csv

# predicst with SAM-like architectures, using point prompts sampled
# from the segmentation model's output logits
python inference.py \
    --output_path "$BASE_PATH/outputs/$EXP_TAG" \
    --sam_model_topology SAMUS \
    --sam_model_path "$BASE_PATH/$SAM_MODEL_PATH" \
    --sam_model_type vit_b \
    --seg_output_path "$BASE_PATH/outputs/$SEG_EXP_TAG" \
    --data_path "$BASE_PATH/data" \
    --data_desc_path "$BASE_PATH/data/metadata/$DATASET_DESCRIPTOR" \
    --split_name val \
    --sampling_step 50 \
    --sampling_mode grid

#python eval.py