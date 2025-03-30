#!/usr/bin/env python3
# move_synthetic_data.py - Script for moving synthetic data to target directory

import os
import shutil
import argparse
from pathlib import Path
import sys

def move_synthetic_data(synthetic_dir, target_dir):
    """
    Moves synthetic data from source directory to target directory,
    maintaining camera folder and ID folder structure.
    
    Args:
        synthetic_dir (str): Path to the synthetic data directory
        target_dir (str): Path to the target directory
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Convert to Path objects for easier handling
    synthetic_path = Path(synthetic_dir)
    target_path = Path(target_dir)
    
    # Check if source directory exists
    if not synthetic_path.exists() or not synthetic_path.is_dir():
        print(f"Error: Source directory '{synthetic_dir}' does not exist!")
        return False
    
    # Create target directory if it doesn't exist
    if not target_path.exists():
        print(f"Target directory '{target_dir}' does not exist. Creating it...")
        target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all camera folders in synthetic directory
    camera_folders = [item for item in synthetic_path.iterdir() if item.is_dir()]
    
    if not camera_folders:
        print(f"No camera folders found in '{synthetic_dir}'!")
        return False
    
    # Process each camera folder
    for cam_folder in camera_folders:
        cam_name = cam_folder.name
        target_cam_folder = target_path / cam_name
        
        print(f"Processing camera folder: {cam_name}")
        
        # Create target camera folder if it doesn't exist
        target_cam_folder.mkdir(exist_ok=True)
        
        # Get all ID folders in the camera folder
        id_folders = [item for item in cam_folder.iterdir() if item.is_dir()]
        
        if not id_folders:
            print(f"No ID folders found in '{cam_folder}'!")
            continue
        
        # Process each ID folder
        for id_folder in id_folders:
            id_name = id_folder.name
            target_id_folder = target_cam_folder / id_name
            
            # Create target ID folder if it doesn't exist
            target_id_folder.mkdir(exist_ok=True)
            
            # Get all image files in the ID folder
            image_files = [f for f in id_folder.glob('*.jpg')]
            
            if not image_files:
                print(f"No images found in {id_folder}")
                continue
            
            # Copy all images to target ID folder
            print(f"Copying {len(image_files)} images from {id_folder} to {target_id_folder}")
            for img_file in image_files:
                target_file = target_id_folder / img_file.name
                shutil.copy2(img_file, target_file)
    
    print("Data movement completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Move synthetic data to target directory')
    
    # Use more explicit argument names with flags
    parser.add_argument('--input', '-i', required=True,
                        help='Path to the input synthetic data directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to the output target directory')
    
    args = parser.parse_args()
    
    result = move_synthetic_data(args.input, args.output)
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())