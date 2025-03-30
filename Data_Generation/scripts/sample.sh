#!/bin/bash
# sample.sh - Script for generating synthetic images using generate_t2i.py

function sample {
    # Default arguments if not provided
    data_dir=${1:-"/root/autodl-tmp/DiVE/synthesize_data/preprocess/step1/2/sysu/sysu"}
    model_path=${2:-"/root/autodl-tmp/autodl-tmp/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"}
    lora_path=${3:-"finetune_model/lora_ti/pytorch_lora_weights.safetensors"}
    embed_path=${4:-"finetune_model/lora_ti/learned_embeds-steps-last.bin"}
    output_dir=${5:-"outputs"}
    resolution=${6:-512}
    
    echo "Starting image generation with:"
    echo "- Data directory: $data_dir"
    echo "- Model path: $model_path"
    echo "- LoRA path: $lora_path" 
    echo "- Embedding path: $embed_path"
    echo "- Output directory: $output_dir"
    echo "- Resolution: $resolution"
    
    # Export the data directory for the Python script to use
    export HUG_LOCAL_IMAGE_TRAIN_DIR="$data_dir"
    
    python generate_t2i.py \
    --output_path "$output_dir" \
    --resolution $resolution \
    --gpu_ids ${GPU_IDS[@]} \
    --cameras cam3 cam6 \
    --repeat_each_id 18 \
    --batch_size 16 \
    --images_per_call 18 \
    --model_path "$model_path" \
    --lora_path "$lora_path" \
    --embed_path "$embed_path" \
    --guidance_scale 7.5
    # Uncomment the line below to run in debug mode (single-process)
    # --debug
}

# To run this script:
#
# 1. Basic usage with default settings:
#    source sample.sh
#    sample
#
# 2. Pass data directory as first parameter:
#    sample "/path/to/data/directory"
#
# 3. Specify all parameters:
#    sample "/path/to/data/directory" "/path/to/model" "/path/to/lora" "/path/to/embed" "custom_outputs" 768
#
# 4. Setting GPU IDs before running:
#    export GPU_IDS="0 1"
#    sample "/path/to/data/directory"
#      
# NOTE ON PARAMETERS:
# 1. data_dir: Path to the training/input data directory (sets HUG_LOCAL_IMAGE_TRAIN_DIR)
# 2. model_path: Path to the base Stable Diffusion model
# 3. lora_path: Path to the LoRA weights file
# 4. embed_path: Path to the textual inversion embeddings
# 5. output_dir: Directory to save generated images
# 6. resolution: Image size in pixels (default: 512)
#
# NOTE ON GPU IDs:
# - The script will use GPU 0 by default if GPU_IDS is not specified
# - When using multiple GPUs, separate IDs with spaces
# - Examples:
#   - For a single GPU: export GPU_IDS="0"
#   - For multiple GPUs: export GPU_IDS="0 1 2 3"
# - The script will distribute generation tasks across all specified GPUs
#
# Parameter explanations for Python script:
# --output_path: Directory to save generated images
# --resolution: Image size in pixels (higher = more detail but requires more GPU memory)
# --gpu_ids: Which GPUs to use for generation
# --cameras: Camera views to generate (controls perspective of images)
# --repeat_each_id: Number of images to generate per class ID per camera
# --batch_size: Number of images processed in parallel per GPU
# --images_per_call: Number of images to generate in each model call
# --model_path: Path to the base Stable Diffusion model
# --lora_path: Path to the LoRA weights file from fine-tuning
# --embed_path: Path to the textual inversion embeddings
# --guidance_scale: Controls how closely generations follow the text prompt (7.5 is standard)

# If this script is called directly (not sourced), run the sample function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sample "$@"
fi