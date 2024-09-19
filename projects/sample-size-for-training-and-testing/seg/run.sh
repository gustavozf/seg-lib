#!/bin/bash

EXP_TAG=seg_model
BASE_PATH=/path/to/input/files
BACKBONE_FILE=pvt_v2_b2.pth
DATASET_DESCRIPTOR=dataset_set_sep_sample.csv

python train.py \
    --model_type CAFE \
    --data_path "$BASE_PATH/data" \
    --output_path "$BASE_PATH/outputs/$EXP_TAG" \
    --checkpoint "$BASE_PATH/pretrained_models/$BACKBONE_FILE" \
    --data_desc_path "metadata/$DATASET_DESCRIPTOR" \
    --input_size 352 \
    --learning_rate 0.0001 \
    --lr_warmup_period 0 \
    --batch_size 20 \
    --epochs 100 \
    --eval_freq 1 \
    --save_freq 1

python eval.py \
    --input_path "$BASE_PATH/outputs/$EXP_TAG" \
    --backbone_weights_path "$BASE_PATH/pretrained_models/$BACKBONE_FILE" \
    --data_path "$BASE_PATH/data" \
    --data_desc_path "metadata/$DATASET_DESCRIPTOR" \
    --batch_size 8 \
    --split_name 'val'
