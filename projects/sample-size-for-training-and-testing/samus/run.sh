#!/bin/bash

EXP_TAG=samus_model
BASE_PATH=/path/to/input/files
BACKBONE_FILE=sam_vit_b_01ec64.pth
DATASET_DESCRIPTOR=dataset_set_sep_sample.csv

python train.py \
    --model_type SAMUS \
    --data_path "$BASE_PATH/data" \
    --output_path "$BASE_PATH/outputs/$EXP_TAG" \
    --data_desc_path "$BASE_PATH/data/metadata/$DATASET_DESCRIPTOR" \
    --checkpoint "$BASE_PATH/pretrained_models/$BACKBONE_FILE" \
    --input_size 256 \
    --embedding_size 128 \
    --learning_rate 0.0001 \
    --lr_warmup_period 0 \
    --batch_size 6 \
    --epochs 100 \
    --eval_freq 1 \
    --save_freq 1 \
    --prompt_type "click" \
    --n_clicks 4 \
    --n_cpus 8

# evaluation using the model + data loader
python eval.py \
    --input_path "$BASE_PATH/outputs/$EXP_TAG" \
    --backbone_weights_path "$BASE_PATH/pretrained_models/$BACKBONE_FILE" \
    --data_path "$BASE_PATH/data" \
    --data_desc_path "$BASE_PATH/data/metadata/$DATASET_DESCRIPTOR" \
    --split_name val \
    --batch_size 8

# evaluation using the predictor class + iterative data loading
python test.py \
    --input_path "$BASE_PATH/outputs/$EXP_TAG" \
    --backbone_weights_path "$BASE_PATH/pretrained_models/$BACKBONE_FILE" \
    --data_path "$BASE_PATH/data" \
    --data_desc_path "$BASE_PATH/data/metadata/$DATASET_DESCRIPTOR" \
    --split_name val \
    --batch_size 8
