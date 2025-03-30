
from typing import Any
import os
import sys
import re
import torch
import argparse
import random
import time
import numpy as np
import pandas as pd

from semantic_aug.datasets.person_revised import PersonHugDataset
from semantic_aug.augmentations.dreabooth_lora_mt import DreamboothLoraGeneration

from tqdm import tqdm
from PIL import Image
from itertools import product
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty

    
def generate_translations(args, in_queue, out_queue, gpu_id):
    model = DreamboothLoraGeneration(
        lora_path=args.lora_path,
        model_path=args.model_path,
        embed_path=args.embed_path,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        mask=args.mask,
        inverted=args.inverted,
        device=f'cuda:{gpu_id}'
    )
    batch_size = args.batch_size

    while True:
        i_lists = []
        cam_names = []
        descriptions = []
        class_ids = []
        strengths = []

        # 1) Pull from queue in mini-batches
        for _ in range(batch_size):
            try:
                # Now the queue items are (i_list, cam_name, desc, class_id, strength)
                q_i_list, q_cam_name, q_desc, q_class_id, q_strength = in_queue.get(timeout=1)
                i_lists.append(q_i_list)
                cam_names.append(q_cam_name)
                descriptions.append(q_desc)
                class_ids.append(q_class_id)
                strengths.append(q_strength)
            except Empty:
                print("Queue empty, exiting.")
                break

        if len(i_lists) == 0:
            break

        # 2) Generate for each item in the batch
        results = []
        for i_list, cam, desc, nm, stg in zip(i_lists, cam_names, descriptions, class_ids, strengths):
            metadata = {
                "camera_name": cam,
                "description": desc,
                "class_id": nm
            }

            # We create a random seed for each image in i_list.
            seeds_for_batch = [
                random.randint(0, 2**32 - 1) for _ in i_list
            ]

            # Generate len(i_list) images in one call
            out_images = model(
                metadata=metadata,
                strength=stg,
                resolution=args.resolution,
                num_images_per_prompt=len(i_list),  # <--- We'll add param support below
                seeds=seeds_for_batch,  # <--- pass the random seeds
            )

            # Save results for later
            results.append( (i_list, cam, desc, nm, stg, out_images) )

        # 3) Save images
        for i_list, cam_name, desc, nm, stg, image_list in results:
            # 'image_list' has len(i_list) images
            for single_i, pil_image in zip(i_list, image_list):
                
                sub_subfolder = f"{int(nm):04d}"
                subfolder = os.path.join(args.output_path, cam_name, sub_subfolder)
                
                os.makedirs(subfolder, exist_ok=True)

            for local_idx, pil_image in enumerate(image_list, start=1):
                filename = f"{local_idx:04d}.jpg"  # always 4-digit zero-padded
                save_path = os.path.join(subfolder, filename)
                pil_image.save(save_path)
                print(f"Saved {save_path}")



def print_instance_attributes(obj):
    """Debug helper."""
    attrs = vars(obj)
    for k, v in attrs.items():
        print(f"{k} = {type(v)} -> {v if len(str(v))<500 else '...'}")
 
def build_queue_via_metadata(
    dataset_obj,
    cameras,
    repeat_each_id,
    threshold,
    in_queue,
    images_per_call=1,  # <--- new param
):
    """
    Build a task queue that:
      1) Iterates cameras from metadata.
      2) Iterates class IDs from dataset_obj.label2class.
      3) Only enqueues classes whose numeric ID > 'threshold'.
      4) Repeats each camera->class combination 'repeat_each_id' times.
      5) Now stores BOTH the camera name and the camera description.

    :param dataset_obj: PersonHugDataset or similar,
                       with .camera_to_description, .label2class, .class2label, etc.
    :param cameras: List of camera names to process (e.g. ['cam1', 'cam2']).
                    If None or empty, uses dataset_obj.camera_to_description.keys().
    :param repeat_each_id: How many times to replicate each camera->class pair.
    :param threshold: Numeric threshold for filtering class IDs (e.g. only > threshold).
    :param in_queue: Queue for multiprocessing tasks.
    """
    # import pdb; pdb.set_trace()
    camera_to_description = dataset_obj.camera_to_description  # e.g. {"cam1": "ABCD1234", ...}                 # e.g. {0: "class_0", 1: "class_1", ...}
    class2label = dataset_obj.class2label 
    # If user doesn't specify cameras, gather them from camera_to_description
    if not cameras:
        cameras = sorted(camera_to_description.keys())

    index_counter = 0
    strength = 1.0

    # Sort class IDs so you process them in ascending order
    all_class_ids = class2label.keys()

    for cam_name in cameras:
        if cam_name not in camera_to_description:
            print(f"[build_queue_via_metadata] Warning: {cam_name} not in camera_to_description!")
            continue

        cam_desc_str = camera_to_description[cam_name]

        for class_id in all_class_ids:
            if int(class_id) <= threshold:
                continue

            # We'll produce 'repeat_each_id' total images for this camera->class combo,
            # but in chunks of size (images_per_call) to reduce pipeline calls.

            start_i = index_counter
            end_i = index_counter + repeat_each_id  # e.g. if repeat_each_id=12, range= [0..12)
            i_values = list(range(start_i, end_i))

            # Chunk them
            for chunk_start in range(start_i, end_i, images_per_call):
                chunk_end = min(chunk_start + images_per_call, end_i)
                i_list = list(range(chunk_start, chunk_end))

                # Enqueue i_list (rather than a single i)
                in_queue.put((i_list, cam_name, cam_desc_str, class_id, strength))

            # Advance overall index_counter by 'repeat_each_id'
            index_counter += repeat_each_id

    print(f"[build_queue_via_metadata] total tasks queued: {index_counter}")
    
    
def main(args):
    """
    An example end-to-end pipeline:
      1. Prepares output dir, seeds RNG.
      2. Loads dataset.
      3. Builds a queue of tasks sorted by camera description.
      4. Spawns worker processes to generate images.
      5. Tracks progress visually via tqdm.
      6. Generates a final CSV (meta.csv) that indexes generated images.
    """

    # 1) Prepare environment
    os.makedirs(args.output_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 2) GPU IDs and queues for multiprocessing
    gpu_ids = args.gpu_ids
    in_queue = Queue()
    out_queue = Queue()

    # 3) Load your dataset
    #    For example, if "sysu" => PersonHugDataset or similar
    #    The user-provided class handles huggingface dataset creation
    train_dataset = PersonHugDataset(split="train", 
                        seed=args.seed, 
                        examples_per_class=args.examples_per_class, 
                        resolution=args.resolution)

    # 4) Build queue by camera/description
    #    Instead of random sampling by labels, we want camera+id-based tasks.
    # Example threshold=1000, repeat_each_id=3, etc.
    build_queue_via_metadata(
        dataset_obj=train_dataset,
        cameras=args.cameras,         # e.g. ['cam1','cam2'] or None
        threshold=args.threshold,     # e.g. 1000
        repeat_each_id=args.repeat_each_id,
        in_queue=in_queue,
        images_per_call=args.images_per_call
    )

    # 5) Debug info: dataset stats
    print_instance_attributes(train_dataset)
    num_classes = len(train_dataset.class_names)
    print(f"[main] Dataset has {len(train_dataset)} items and {num_classes} classes.")

    total_tasks = in_queue.qsize()
    print('[main] Number of total tasks:', total_tasks)

    # 6) Debug vs. Normal mode
    if args.debug:
        print("debug run")
        # Single-process run
        generate_translations(args, in_queue, out_queue, gpu_id=gpu_ids[0])

    else:
        print("multi-process run")
        processes = []
        with tqdm(total=total_tasks, desc="Processing") as pbar:
            for process_id, gpu_id_value in enumerate(gpu_ids):
                p = Process(
                    target=generate_translations,
                    args=(args, in_queue, out_queue, gpu_id_value)
                )
                p.start()
                processes.append(p)

            # Update progress bar while processes are alive
            while any(p.is_alive() for p in processes):
                current_queue_size = in_queue.qsize()
                processed = total_tasks - current_queue_size
                pbar.n = processed
                pbar.refresh()
                time.sleep(1)

            for p in processes:
                p.join()
                
    print("All processes finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument("--output_path", type=str, default="outputs", help="Root path for output data.")
    parser.add_argument("--confusion_matrix_path", type=str, default=None, help="Path to a precomputed confusion matrix.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--examples_per_class", type=int, default=-1, help="Number of examples per class (if applicable).")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--prompt", type=str, default="a {description} photo of a {name}", help="Base prompt string.")
    parser.add_argument("--mask", type=int, default=0, choices=[0, 1], help="Whether to apply a mask (0=no, 1=yes).")
    parser.add_argument("--inverted", type=int, default=0, choices=[0, 1], help="Whether to invert the mask (0=no, 1=yes).")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="List of GPU IDs to use.")
    parser.add_argument("--debug", action="store_true", help="Use debugging mode (single-process).")
    parser.add_argument("--cameras", nargs="+", default=None, help="List of camera names or descriptions to process (e.g. cam1 cam2). If not set, we gather all unique cameras from the dataset.")
    parser.add_argument("--repeat_each_id", type=int, default=1, help="How many times to enqueue each ID in every camera.")
    parser.add_argument("--threshold", type=int, default=533, help="Ignore classes with numeric ID <= this threshold.")
    parser.add_argument("--images_per_call", type=int, default=1, help="Number of images generated per pipeline call.")
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Base model path.")
    parser.add_argument("--lora_path", type=str, default='finetune_model/lora_ti/pytorch_lora_weights.safetensors', help="path to lora")
    parser.add_argument("--embed_path", type=str, default='finetune_model/lora_ti/learned_embeds-steps-last.bin', help="path to textual embedding")
    parser.add_argument("--guidance_scale",type=float,default=7.5,help="Guidance scale for classifier-free guidance (how strictly to follow the text prompt, higher = more strict)",)
    args = parser.parse_args()

    # Please check the corresponding ckpt path before sampling !!! 

    torch.multiprocessing.set_start_method('spawn')
    os.makedirs(args.output_path, exist_ok=True)
    
    main(args)