#!/bin/bash

# 定义参数
DATASET="llcm"
DATA_PATH="/root/autodl-tmp/deen_reid/LLCM/DEEN/llcm_orin/"
GPU="0"
MODEL_PATH="save_model_llcm_orin/"

LOG_PATH="log_llcm_orin/"
N_CLASS="1724"
MODE="all"
TVSEARCH="True"
RESUME="llcm_deen_p4_n16_lr_0.1_seed_0_best.t"  # 从检查点恢复的文件名

# 训练模型
python train.py \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --model_path $MODEL_PATH \
  --log_path $LOG_PATH

# 测试模型：v2i 模式
python test.py \
  --dataset $DATASET \
  --mode $MODE \
  --tvsearch $TVSEARCH \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --n_class $N_CLASS \
  --test_mode v2i \
  --resume $RESUME

# 测试模型：i2v 模式
python test.py \
  --dataset $DATASET \
  --mode $MODE \
  --tvsearch $TVSEARCH \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --n_class $N_CLASS \
  --test_mode i2v \
  --resume $RESUME