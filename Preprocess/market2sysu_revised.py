#!/usr/bin/env python3

import os
import shutil
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Convert Market-1501 dataset to SYSU-MM01 format")

    # Hyperparameters:
    parser.add_argument(
        "--market_gt_bbox_dir",
        type=str,
        default="/root/autodl-tmp/DiVE/synthesize_data/preprocess/step1/2/market_data/Market-1501-v15.09.15/gt_bbox",
        help="Path of the gt_bbox folder in the Market-1501 dataset"
    )
    parser.add_argument(
        "--sysu_dir",
        type=str,
        default="sysu_filtered",
        help="Root directory path of the SYSU-MM01 dataset"
    )
    parser.add_argument(
        "--max_sysu_id",
        type=int,
        default=533,
        help="Current maximum SYSU ID (default 533, can be changed dynamically)"
    )
    parser.add_argument(
        "--min_images_per_id",
        type=int,
        default=12,
        help="Threshold for minimum number of images per ID (default 12)"
    )

    args = parser.parse_args()

    # Market-1501 gt_bbox folder path
    market_gt_bbox_dir = args.market_gt_bbox_dir

    # SYSU-MM01 root directory path
    sysu_dir = args.sysu_dir

    # Name of the new camera folder
    new_cam_folder = "cam7"
    new_cam_dir = os.path.join(sysu_dir, new_cam_folder)

    # Create the new camera folder (if it does not exist)
    os.makedirs(new_cam_dir, exist_ok=True)

    # Retrieve the current max SYSU ID
    max_sysu_id = args.max_sysu_id

    # Define the threshold for the number of images
    min_images_per_id = args.min_images_per_id

    # 1. Collect all image files corresponding to each old_id
    id_to_images = {}  # key: market_id, value: list of image filenames

    all_files = os.listdir(market_gt_bbox_dir)
    # Optional: sort files here, mainly for consistent traversal; 
    # we will still do a separate sort of IDs later when assigning them
    all_files = sorted(all_files)

    for image_file in all_files:
        # e.g., 0022_c6s1_003951_00.jpg
        # Extract the old_id from the first 4 characters if they are digits
        if image_file[:4].isdigit():
            market_id = int(image_file[:4])
        else:
            print(f"Skipping file that does not meet the format: {image_file}")
            continue
        
        if market_id not in id_to_images:
            id_to_images[market_id] = []
        id_to_images[market_id].append(image_file)

    # 2. Filter out old_ids that meet min_images_per_id
    selected_market_ids = []
    for market_id, images_list in id_to_images.items():
        if len(images_list) >= min_images_per_id:
            selected_market_ids.append(market_id)

    # 3. Sort selected_market_ids as needed (here: ascending order)
    selected_market_ids.sort()

    # 4. Assign new IDs and copy files
    id_mapping = {}
    for i, market_id in enumerate(selected_market_ids, start=1):
        new_sysu_id = max_sysu_id + i  # Assign IDs sequentially
        id_mapping[market_id] = new_sysu_id
        
        # Create a new folder for this ID
        new_id_dir = os.path.join(new_cam_dir, f"{new_sysu_id:04d}")
        os.makedirs(new_id_dir, exist_ok=True)
        
        # Copy all images of this market_id and rename them in order
        images_list = id_to_images[market_id]
        images_list.sort()  # You can also sort according to your needs
        for idx, image_file in enumerate(images_list, start=1):
            old_image_path = os.path.join(market_gt_bbox_dir, image_file)
            new_image_path = os.path.join(new_id_dir, f"{idx:04d}.jpg")
            shutil.copy(old_image_path, new_image_path)

    # 5. Update train_id.txt
    train_id_file = os.path.join(sysu_dir, "exp", "train_id.txt")
    with open(train_id_file, "r") as f:
        train_ids = f.read().strip().split(",")

    # Append the newly added IDs to the train_id list
    new_train_ids = train_ids + [str(max_sysu_id + i) for i in range(1, len(selected_market_ids) + 1)]

    with open(train_id_file, "w") as f:
        f.write(",".join(new_train_ids))

    print(f"{len(selected_market_ids)} Market-1501 IDs that meet the conditions have been converted to the SYSU-MM01 format and added to train_id.txt.")

    # 6. Save the mapping of old and new IDs to a JSON file with annotations
    mapping_save_dir = os.path.join(sysu_dir, "exp")
    os.makedirs(mapping_save_dir, exist_ok=True)
    
    # Create the JSON mapping file
    mapping_file = os.path.join(mapping_save_dir, "id_mapping.json")
    
    # Prepare the JSON data structure
    mapping_data = {
        "title": "ID Mapping from Market-1501 to SYSU-MM01",
        "annotations": {
            "original_id": "Market-1501 ID",
            "new_id": "SYSU-MM01 ID"
        },
        "mappings": {str(old_id): new_id for old_id, new_id in id_mapping.items()},
        "total_mappings": len(id_mapping)
    }
    
    # Write the JSON file with proper formatting
    with open(mapping_file, "w") as f:
        json.dump(mapping_data, f, indent=4)

    print(f"The mapping of old and new IDs has been saved to {mapping_file}.")

if __name__ == "__main__":
    main()