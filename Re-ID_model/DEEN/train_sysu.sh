#!/bin/bash

# Variables
DATASET="sysu"
DATA_PATH="path/to/synthetic_sysu/"
GPU="0"
MODEL_PATH="save_model_sysu/"
LOG_PATH="log_sysu_orin/"

# train
python train.py \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --model_path $MODEL_PATH \
  --log_path $LOG_PATH