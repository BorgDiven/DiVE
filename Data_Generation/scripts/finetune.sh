#!/bin/bash
# finetune.sh - Script for finetuning a text-to-image model with LoRA and textual inversion

function finetune {
    # Default arguments if not provided
    data_dir=${1:-"/root/autodl-tmp/DiVE/synthesize_data/preprocess/step1/2/sysu/sysu"}
    model_path=${2:-"/root/autodl-tmp/autodl-tmp/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"}
    output_dir=${3:-"finetune_model/lora_ti"}
    max_steps=${4:-400000}
    batchsize=${5:-16}
    nepoch=${6:-100}
    rank=${7:-128}
    checkpointing_steps=${8:-40000}
    
    # Get GPU IDs - allow it to be passed as 9th parameter or use environment variable
    gpu_ids=${9:-${GPU_IDS:-0}}
    
    # Set environment variables
    export MODEL_NAME="$model_path"
    export HUG_LOCAL_IMAGE_TRAIN_DIR="$data_dir"

    script="train_t2i_lora_ti.py"

    echo "Starting finetuning with:"
    echo "- Data directory: $data_dir"
    echo "- Model path: $model_path"
    echo "- Output directory: $output_dir"
    echo "- Max steps: $max_steps"
    echo "- Batch size: $batchsize"
    echo "- Number of epochs: $nepoch"
    echo "- LoRA rank: $rank"
    echo "- Checkpointing steps: $checkpointing_steps"
    echo "- GPU IDs: $gpu_ids"

    # Create GPU configuration for accelerate
    gpu_args=""
    if [ -n "$gpu_ids" ]; then
        # Convert space-separated GPU IDs to comma-separated for accelerate
        gpu_args="--gpu_ids=$(echo $gpu_ids | tr ' ' ',')"
    fi

    command="accelerate launch $gpu_args --mixed_precision='fp16' --main_process_port 29507 '$script' \
        --pretrained_model_name_or_path='$MODEL_NAME' \
        --caption_column='text' \
        --random_flip \
        --max_train_steps='$max_steps' \
        --num_train_epochs=$nepoch --checkpointing_steps=$checkpointing_steps \
        --learning_rate=5e-05 \
        --lr_scheduler='constant' --lr_warmup_steps=0 \
        --seed=42 \
        --rank=$rank \
        --local_files_only \
        --examples_per_class -1 \
        --train_batch_size $batchsize \
        --output_dir='$output_dir' \
        --report_to='tensorboard'"

    eval "$command"
}

# To run this script:
#
# 1. Basic usage with default settings:
#    source finetune.sh
#    finetune
#
# 2. Specify data directory and model:
#    finetune "/path/to/data" "/path/to/model"
#
# 3. Specify all major parameters:
#    finetune "/path/to/data" "/path/to/model" "output_dir" 500000 32 200 256 50000
#
# 4. Set GPU IDs via parameter (9th position):
#    finetune "/path/to/data" "/path/to/model" "output_dir" 400000 16 100 128 40000 "0 1 2 3"
#
# 5. Set GPU IDs via environment variable:
#    export GPU_IDS="0 1"
#    finetune "/path/to/data"
#      
# NOTE ON GPU IDs:
# - You can control GPU IDs in three ways:
#   1. Pass as the 9th parameter to the finetune function
#   2. Set the GPU_IDS environment variable before calling finetune
#   3. If neither is provided, it defaults to GPU 0
#
# - For multiple GPUs, use space-separated numbers, e.g., "0 1 2 3"
#   (they will be converted to comma-separated format for accelerate)
#
# Parameter explanations:
# 1. data_dir: Path to the training/input data directory
# 2. model_path: Path to the base Stable Diffusion model
# 3. output_dir: Directory to save finetuned model
# 4. max_steps: Maximum number of training steps
# 5. batchsize: Number of images per batch
# 6. nepoch: Number of training epochs
# 7. rank: LoRA rank (higher = more capacity but uses more memory)
# 8. checkpointing_steps: How often to save checkpoints
# 9. gpu_ids: Space-separated list of GPU IDs to use

# If this script is called directly (not sourced), run the finetune function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    finetune "$@"
fi