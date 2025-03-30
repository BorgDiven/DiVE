import argparse
import os
import random
import string
import json
from PIL import Image
import datasets
from datasets import Dataset, Features, Image as ImageFeature

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a dataset from images and store metadata in dataset.info.description."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Name/identifier of the dataset (e.g. 'sysu').")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the input dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to store the final dataset.")
    parser.add_argument("--num_cameras", type=int, required=True, help="Number of cameras in the dataset.")
    return parser.parse_args()

def read_ids(file_path):
    with open(file_path, "r") as f:
        return f.read().strip().split(',')

def generate_unique_descriptions(count):
    """
    Generate 'count' random codes of length 4 with no character overlap among them.
    If count > 15, it will raise a ValueError, because we only have 62 distinct
    alphanumeric characters in total.
    """
    all_chars = list(string.ascii_letters + string.digits)  # 62 unique chars
    random.shuffle(all_chars)  # randomize the pool of characters

    max_codes = len(all_chars) // 4  # each code needs 4 unique chars
    if count > max_codes:
        raise ValueError(
            f"Cannot generate {count} completely different codes of length 4. "
            f"Maximum possible is {max_codes}."
        )

    codes = []
    start = 0
    for _ in range(count):
        # Take 4 characters without overlap
        chunk = all_chars[start : start + 4]
        codes.append("".join(chunk))
        start += 4

    return codes

def main():
    args = parse_args()

    # Extract arguments
    current_dataset = args.dataset
    base_path = args.base_path
    output_path = args.output_path
    num_cameras = args.num_cameras

    # Paths
    train_id_path = os.path.join(base_path, "exp", "train_id.txt")
    val_id_path = os.path.join(base_path, "exp", "val_id.txt")

    # Read IDs
    train_ids = read_ids(train_id_path)
    val_ids = read_ids(val_id_path)
    all_ids = sorted(train_ids + val_ids, key=int)

    # Create label mapping
    id_to_label = {id_str: i for i, id_str in enumerate(all_ids)}

    # Prepare data
    data = {"image": [], "label": [], "description": [], "camera": []}

    # Generate per-camera code descriptions
    camera_descriptions = generate_unique_descriptions(num_cameras)

    # Create mappings
    camera_to_description = {f"cam{i}": desc for i, desc in enumerate(camera_descriptions, 1)}
    description_to_camera = {desc: f"cam{i}" for i, desc in enumerate(camera_descriptions, 1)}

    # Optionally save these mappings to a file
    # You can comment this out if not needed
    camera_mappings = {
        "dataset": current_dataset,
        "num_cameras": num_cameras,
        "camera_to_description": camera_to_description,
        "description_to_camera": description_to_camera
    }
    mapping_file = f"camera_description_mappings_{current_dataset}.json"
    with open(mapping_file, "w") as f:
        json.dump(camera_mappings, f)

    def process_images(camera_path, camera_name):
        description = camera_to_description[camera_name]
        for id_str in all_ids:
            formatted_id = f"{int(id_str):04d}"
            id_path = os.path.join(camera_path, formatted_id)
            if os.path.exists(id_path):
                for img_name in os.listdir(id_path):
                    img_path = os.path.join(id_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            if height >= 2 * width:
                                data["image"].append(img.convert("RGB"))
                                data["label"].append(id_to_label[id_str])
                                data["description"].append(description)
                                data["camera"].append(camera_name)
                            else:
                                print(f"Skipping image {img_path} due to insufficient height")
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")

    # Process images by camera
    camera_paths = [os.path.join(base_path, f"cam{i}") for i in range(1, num_cameras + 1)]
    for i, camera_path in enumerate(camera_paths, 1):
        camera_name = f"cam{i}"
        process_images(camera_path, camera_name)

    # Build the dataset
    dataset = Dataset.from_dict(
        data,
        features=datasets.Features(
            {
                "image": ImageFeature(),
                "label": datasets.ClassLabel(names=all_ids),
                "description": datasets.Value("string"),
                "camera": datasets.Value("string"),
            }
        )
    )

    # Create index mappings for quick retrieval by camera
    camera_indices = {}
    for cam_name in camera_to_description.keys():
        camera_indices[cam_name] = [i for i, c in enumerate(data["camera"]) if c == cam_name]

    # Prepare metadata dict
    metadata_dict = {
        "dataset": current_dataset,
        "num_cameras": num_cameras,
        "camera_indices": camera_indices,
        "camera_to_description": camera_to_description,
        "description_to_camera": description_to_camera,
    }

    # Store metadata as JSON in dataset.info.description
    metadata_json = json.dumps(metadata_dict, indent=2)
    dataset.info.description = f"Custom metadata:\n{metadata_json}"

    # Create the full output path by combining output_path and dataset name
    full_output_path = os.path.join(output_path, current_dataset)

    # Create directory if it doesn't exist
    os.makedirs(full_output_path, exist_ok=True)

    # Save the dataset to the combined path
    dataset.save_to_disk(full_output_path)

    # Print summary
    print(f"Dataset '{current_dataset}' created with {len(dataset)} images.")
    print(f"Number of cameras: {num_cameras}")
    print(f"Unique camera descriptions created: {len(set(data['description']))}")
    print(f"Dataset saved to: {full_output_path}")

if __name__ == "__main__":
    main()