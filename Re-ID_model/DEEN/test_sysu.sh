#!/bin/bash

# variables
DATASET="sysu"
DATA_PATH="path/to/synthetic_sysu/" #add / at the end ! ! !
GPU="0"
MODEL_PATH="save_model_sysu/"
N_CLASS="1406"
TVSEARCH="True"
RESUME="sysu_deen_p4_n16_lr_0.1_seed_0_best.t"  

# test：MODE="all"
python test.py \
  --dataset $DATASET \
  --mode "all" \
  --tvsearch $TVSEARCH \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --n_class $N_CLASS \
  --resume $RESUME

# test：MODE="indoor"
python test.py \
  --dataset $DATASET \
  --mode "indoor" \
  --tvsearch $TVSEARCH \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --gpu $GPU \
  --n_class $N_CLASS \
  --resume $RESUME