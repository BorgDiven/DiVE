#!/bin/bash

# 定义参数
DATASET="regdb"
DATA_PATH="/root/autodl-tmp/deen_reid/LLCM/DEEN/regdb_orin/"
GPU="0"
MODEL_PATH="save_model_regdb_orin/"
LOG_PATH="log_regdb_orin/"
N_CLASS="1217"
MODE="all"
RESUME="regdb_deen_p4_n16_lr_0.1_seed_0_best.t"  # 从检查点恢复的文件名

# 训练模型
python train.py \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --model_path $MODEL_PATH \
  --log_path $LOG_PATH

# 测试模型：TVSEARCH="True"
python test.py \
  --dataset $DATASET \
  --mode $MODE \
  --tvsearch "True" \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --n_class $N_CLASS \
  --resume $RESUME

# 测试模型：TVSEARCH="False"
python test.py \
  --dataset $DATASET \
  --mode $MODE \
  --tvsearch "False" \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --n_class $N_CLASS \
  --resume $RESUME