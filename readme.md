# DiVE

This repository contains the official implementation of our paper "Diffusion-based Synthetic Data Generation for Visible-Infrared Person Re-Identification" accepted at AAAI 2025.

## Environment Setup

1. **Clone** this repository:

   ```bash
   git clone https://github.com/BorgDiven/DiVE.git
   cd DiVE
   ```

2. **Create** a conda environment using the provided `environment.yml` file:

   **Please follow versions pinned in the environment.yml file! (My cuda version is 12.1.)**

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate** your newly created environment:

   ```bash
   conda activate <env_name>
   ```

## Preprocess

1. **Download the SYSU and Market-1501 datasets**


2. **Run `market2sysu_revised.py`** (**cd Preprocess**) 
   Convert the Market dataset into SYSU format (Market as one camera) and add Market into the SYSU dataset:

   ```bash
   python market2sysu.py \
       --market_gt_bbox_dir /path/to/market_data/Market-1501-v15.09.15/gt_bbox \
       --sysu_dir /path/to/sysu_original_data \
       --max_sysu_id 533 \
       --min_images_per_id 12
   ```

   **Parameters to modify**:
   - **market_gt_bbox_dir**: `gt_bbox` directory in the Market-1501 dataset  
   - **sysu_dir**: Path to the SYSU-MM01 dataset  

   After running this, the Market data will be added to the SYSU data folder.

3. **Run `c2hf_revised.py`** (**cd Preprocess**) 
   Convert the SYSU+Market dataset into a format ready for training:

   ```bash
   python c2hf_revised.py \
       --dataset sysu \
       --base_path /path/to/sysu_market \
       --output_path /path/to/sysu_market_huggingface \
       --num_cameras 7
   ```

   **Parameters to modify**:
   - **base_path**: Path of the SYSU+Market data  
   - **output_path**: Path of the processed SYSU+Market data, compatible with Diffuser training


## Data Generation

1. **Download SD-1.5 model Using Hugging Face CLI**

   Install the Hugging Face Hub CLI:

   ```bash
   pip install huggingface_hub
   ```

   Log in to Hugging Face:

   ```bash
   huggingface-cli login
   ```
   Go to https://huggingface.co/settings/tokens, generate a new personal access token, and use it when prompted by huggingface-cli login.

   Download a model:

   ```bash
   huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir ./stable-diffusion-v1-5
   Attention: add the "--local-dir-use-symlinks False" to directly download to specific path
   ```

Change directory to: cd Data_Generation

1. **Finetune the SD model** (**cd Data_Generation**) 
   Train the LoRA branch and textual embedding.

   ```bash
   cd Data_Generation
   source scripts/finetune.sh 
   export GPU_IDS=(0)
   finetune "/path/to/processed_data" "/path/to/sd_model"
   ```

   **Parameters to customize**:  
   - `"/path/to/processed_data"`: data generated by **c2hf_revised.py**  
   - `"/path/to/sd_model"`: the local SD model weights (e.g., `models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9`)

2. **Generate the data** (**cd Data_Generation**) 
   Generate the infrared synthetic data.

   ```bash
   source scripts/sample.sh 
   export GPU_IDS=(0)
   sample "/path/to/processed_data" "/path/to/sd_model"
   ```

   **Parameters to customize**:  
   - `"/path/to/processed_data"`: data generated by **c2hf_revised.py**  
   - `"/path/to/sd_model"`: the local SD model weights (e.g., `models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9`)

3. **Add the generated data into SYSU-Market data** (**cd Preprocess**) 

   ```bash
   python move_synthetic_data.py -i /path/to/synthetic_data -o /path/to/synthetic_data
   ```


## Re-ID Training
Train SYSU data with DEEN method

1. **Pre-process the dataset**
   ```bash
    cd Re-ID_model/DEEN
    python pre_process_sysu.py
   ```

2. **Train the Re-ID model**
    
    Set the sh file:

   ```bash
    DATA_PATH="path/to/synthetic_sysu/" #add / at the end ! ! !
   ```

    Run training:

   ```bash
    sh train.sh
   ```

3. **Test the Re-ID model**
    
    Set the sh file:

   ```bash
    DATA_PATH="path/to/synthetic_sysu/" #add / at the end ! ! !
    RESUME="checkpoint_name" #sysu_deen_p4_n16_lr_0.1_seed_0_best.t
    N_CLASS="1406" # 395(sysu ids) + 1011(selected market ids, check it at synthetic_data/exp/id_mapping.json) 
   ```

    Run testing:

   ```bash
    sh test.sh
   ```
