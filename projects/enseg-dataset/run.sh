#!/bin/bash

DATA_PATH=/path/to/data/labelme/format
OUTPUT_PATH=/path/to/output
SAM_MODELS_PATH=/path/to/sam/models

echo "Formatting the data for YOLO training..."
categories=("2C" "4C" "5C" "22TW" "23TW" "28TW")
for category in "${categories[@]}"; do
    echo "Processing ${category}..."
    python ../../tools/labelme_2_yolov8.py \
        --input-path "${DATA_PATH}/${category}" \
        --output-path "${OUTPUT_PATH}/yolo_data/${category}" \
        --mode seg
done

python train_yolo_seg.py \
    --data_path $OUTPUT_PATH/yolo_data/ \
    --output_path $OUTPUT_PATH/yolo_seg/ \
    --epochs 100 \
    --train_batch_size -1 \
    --val_batch_size 2 \
    --image_size 640

python evaluate_sam.py \
    --data_path $DATA_PATH \
    --models_path $SAM_MODELS_PATH \
    --output_path $OUTPUT_PATH/sam/

python evaluate_fusion.py \
    --data_path $DATA_PATH \
    --yolo_model_path $OUTPUT_PATH/yolo_seg/yolov8m \
    --sam_model_path $SAM_MODELS_PATH/sam_vit_b_01ec64.pth \
    --output_path $OUTPUT_PATH/fusion/