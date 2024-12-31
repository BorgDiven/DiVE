#!/bin/bash

# 定义参数
DATASET="sysu"
DATA_PATH="/root/autodl-tmp/deen_reid/LLCM/DEEN/sysu_orin/"
GPU="0"
MODEL_PATH="save_model_sysu_orin/"
LOG_PATH="log_sysu_orin/"

# 前台运行命令并将输出重定向到日志文件
python train.py \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --model_path $MODEL_PATH \
  --log_path $LOG_PATH \