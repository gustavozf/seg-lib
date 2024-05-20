#!/bin/bash

EXP_TAG=v1_1_0
BASE_PATH=/nfsd/iaslab4/Users/zanonif/project
BACKBONE_FILE=pvt_v2_b2.pth
DATASET_DESCRIPTOR=dataset_full.csv

python train.py \
    --model_type PTVv2 \
    --data_path "$BASE_PATH/data/train" \
    --output_path "$BASE_PATH/outputs/train/$EXP_TAG" \
    --data_desc_path "metadata/$DATASET_DESCRIPTOR" \
    --checkpoint "$BASE_PATH/pretrained_models/$BACKBONE_FILE" \
    --input_size 352 \
    --learning_rate 0.0001 \
    --lr_warmup_period 0 \
    --batch_size 20 \
    --epochs 100 \
    --eval_freq 1 \
    --save_freq 1

python eval.py \
    --input_path "$BASE_PATH/outputs/train/$EXP_TAG" \
    --backbone_weights_path "$BASE_PATH/pretrained_models/$BACKBONE_FILE" \
    --data_path "$BASE_PATH/data/test" \
    --data_desc_path "metadata/$DATASET_DESCRIPTOR" \
    --batch_size 8
