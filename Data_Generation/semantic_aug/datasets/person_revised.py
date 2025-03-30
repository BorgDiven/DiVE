# import sys
import os
import random
import torch
# sys.path.append('/data/zhicai/code/Diff-Mix/')
from semantic_aug.few_shot_dataset import HugFewShotDataset
from semantic_aug.datasets.utils import IMAGENET_TEMPLATES_DIVE
# from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict
# from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from shutil import copyfile
from PIL import Image, ImageDraw
from collections import defaultdict
from datasets import load_dataset, load_from_disk

SUPER_CLASS_NAME = 'person'
# HUG_LOCAL_IMAGE_TRAIN_DIR = "/root/autodl-tmp/process_sysu/person_sysu_market_ca_12_1011"
HUG_LOCAL_IMAGE_TEST_DIR = "/root" #not used, doesn't matter

HUG_LOCAL_IMAGE_TRAIN_DIR = os.getenv("HUG_LOCAL_IMAGE_TRAIN_DIR")
if HUG_LOCAL_IMAGE_TRAIN_DIR is None:
    raise ValueError("HUG_LOCAL_IMAGE_TRAIN_DIR env variable is not set.")


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class PersonHugDataset(HugFewShotDataset):
    super_class_name = SUPER_CLASS_NAME

    def __init__(self, *args, 
                 split: str = "train", 
                 seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 image_test_dir: str = HUG_LOCAL_IMAGE_TEST_DIR, 
                 examples_per_class: int = -1, 
                 synthetic_probability: float = 0.5,
                 return_onehot: bool = False,
                 soft_scaler: float = 0.9,
                 synthetic_dir: str = None,
                 image_size: int = 512,
                 crop_size: int = 448,
                 corrupt_prob=0.0,
                 **kwargs):

        # Choose dataset path based on split
        if split == 'train':
            dataset = load_from_disk(image_train_dir)
        else:
            dataset = load_from_disk(image_test_dir)

        # -------------------------------------------
        # (1) Parse Custom Metadata from info.description
        # -------------------------------------------
        import json

        desc_text = dataset.info.description or ""
        lines = desc_text.split("\n", 1)
        if len(lines) > 1 and lines[0].startswith("Custom metadata:"):
            # The second part lines[1] is presumably the JSON
            metadata_json = lines[1]
            try:
                metadata_dict = json.loads(metadata_json)
            except json.JSONDecodeError as e:
                print("[PersonHugDataset] Warning: Could not decode JSON metadata:", e)
                metadata_dict = {}
        else:
            metadata_dict = {}

        # Store it in self so we can access later
        self.metadata_dict = metadata_dict
        # If present, these might map cameras -> descriptions or cameras -> indices
        self.camera_to_description = metadata_dict.get("camera_to_description", {})
        self.description_to_camera = metadata_dict.get("description_to_camera", {})
        self.camera_indices = metadata_dict.get("camera_indices", {})

        # -------------------------------------------
        # (2) Standard RNG, etc.
        # -------------------------------------------
        random.seed(seed)
        np.random.seed(seed)

        # -------------------------------------------
        # (3) Optional: Subsample dataset to [examples_per_class]
        # -------------------------------------------
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset['label']
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population, fallback to random.choices")
                    sampled_indices = random.choices(items, k=examples_per_class)
                _all_indices.extend(sampled_indices)

            dataset = dataset.select(_all_indices)

        # -------------------------------------------
        # (4) Save dataset in self
        # -------------------------------------------
        self.dataset = dataset
        self.class_names = [
            name.replace('/', ' ')
            for name in dataset.features['label'].names
        ]
        self.num_classes = len(self.class_names)

        # -------------------------------------------
        # (5) Optional: Corrupt some labels
        # -------------------------------------------
        if corrupt_prob > 0:
            print(f"[PersonHugDataset] corrupt_prob: {corrupt_prob}")
            self.corrupt_labels(corrupt_prob)

        # Create label <-> class dict
        class2label = self.dataset.features['label']._str2int
        self.class2label = {k.replace('/', ' '): v for k, v in class2label.items()}
        self.label2class = {v: k.replace('/', ' ') for k, v in class2label.items()}

        # Keep track of indices for each numeric label
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)

        # -------------------------------------------
        # (6) Call Parent Constructor
        # -------------------------------------------
        super().__init__(
            *args,
            split=split,
            examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            return_onehot=return_onehot,
            soft_scaler=soft_scaler,
            synthetic_dir=synthetic_dir,
            image_size=image_size,
            crop_size=crop_size,
            **kwargs
        )

    # ---------------------------------------------------------------
    #  (7) Corrupt labels if requested
    # ---------------------------------------------------------------
    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.dataset['label'])
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(len(self.class_names), mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.dataset = (
            self.dataset
                .remove_columns("label")
                .add_column("label", labels)
                .cast(self.dataset.features)
        )

    # ---------------------------------------------------------------
    #  (8) Basic dataset protocol
    # ---------------------------------------------------------------
    def __len__(self):
        return len(self.dataset)

    def get_image_by_idx(self, idx: int) -> Image.Image:
        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:
        return self.dataset[idx]['label']

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(
            name=self.label2class[self.get_label_by_idx(idx)],
            super_class=self.super_class_name
        )

class PersonHugDatasetForT2I(torch.utils.data.Dataset):

    super_class_name = SUPER_CLASS_NAME

    def __init__(self, *args, split: str = "train",
                 seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 max_train_samples: int = -1,
                 class_prompts_ratio: float = 0.5,
                 resolution: Tuple[int, int] = (512, 256),#int = 512,
                 center_crop: bool = False,
                 random_flip: bool = False,
                 use_placeholder: bool = False,
                 examples_per_class: int = -1,
                 **kwargs):

        super().__init__()    

        # dataset = load_dataset(image_train_dir, split='train')
        dataset = load_from_disk(image_train_dir)

        random.seed(seed)
        np.random.seed(seed)
        if max_train_samples is not None and max_train_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(max_train_samples))
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset['label']
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population or is negative, use random.choices instead")
                    sampled_indices = random.choices(items, k=examples_per_class)
                    
                label_to_indices[key] = sampled_indices 
                _all_indices.extend(sampled_indices)
            dataset = dataset.select(_all_indices)
        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]
        self.dataset = dataset
        self.class2label = self.dataset.features['label']._str2int
        self.label2class = {v: k for k, v in self.class2label.items()} 
        self.num_classes = len(self.class_names)
        self.class_prompts_ratio = class_prompts_ratio
        self.use_placeholder = use_placeholder        
        self.name2placeholder = None
        self.placeholder2name = None
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)
            
        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        import json

        desc_text = dataset.info.description or ""
        lines = desc_text.split("\n", 1)
        if len(lines) > 1 and lines[0].startswith("Custom metadata:"):
            # The second part lines[1] is presumably the JSON
            metadata_json = lines[1]
            try:
                metadata_dict = json.loads(metadata_json)
            except json.JSONDecodeError as e:
                print("[PersonHugDataset] Warning: Could not decode JSON metadata:", e)
                metadata_dict = {}
        else:
            metadata_dict = {}

        # Store it in self so we can access later
        self.metadata_dict = metadata_dict
        # If present, these might map cameras -> descriptions or cameras -> indices
        self.camera_to_description = metadata_dict.get("camera_to_description", {})
        self.description_to_camera = metadata_dict.get("description_to_camera", {})
        self.camera_indices = metadata_dict.get("camera_indices", {})


    def __len__(self):
        
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # import pdb;pdb.set_trace()
        image = self.get_image_by_idx(idx)
        prompt = self.get_prompt_by_idx(idx)
        # label = self.get_label_by_idx(idx)

        return dict(pixel_values=self.transform(image), caption = prompt)
    
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.dataset[idx]['label']

    def get_prompt_by_idx(self, idx: int) -> str:
        """
        Return a string prompt for the given index,
        using the updated IMAGENET_TEMPLATES_SMALL
        which has two placeholders: 
        - First for description 
        - Second for name
        """

        # (A) Decide if we build a prompt from label or from random lines in description
        #     Right now, we always do the "class prompt" path if random.random() < self.class_prompts_ratio,
        #     else we pick a random line from the description. 
        #     (Change that ratio or logic as desired.)
        flag = random.random() < self.class_prompts_ratio

        # (B) Access the textual description from the dataset
        domain_description = self.dataset[idx]['description'].strip()
        
        if flag:
            # import pdb;pdb.set_trace()
            # (C) Optionally get the placeholder name vs. direct class label
            if self.use_placeholder and self.name2placeholder is not None:
                content = self.name2placeholder[self.label2class[self.dataset[idx]['label']]]
            else:
                content = self.label2class[self.dataset[idx]['label']]

            # Append the "super_class_name" if you want, e.g. "person"
            content += " person"

            # (D) Choose a template. Now each template has form "a {} photo of a {}", etc.
            template = random.choice(IMAGENET_TEMPLATES_DIVE)

            # (E) Fill in two placeholders:
            #     1) the domain description
            #     2) the subject content
            prompt = template.format(domain_description, content)
        else:
            # If not using the "class prompt" approach, pick a random line from description
            # e.g. in case the user's 'description' has multiple lines
            lines = domain_description.split('\n')
            prompt = random.choice(lines) if lines else domain_description

        return prompt

    # def get_prompt_by_idx(self, idx: int) -> int:
        
    #     # import pdb;pdb.set_trace()
    #     # randomly choose from class name or description
    #     flag = random.random() < 1 #self.class_prompts_ratio 
    #     if flag:
    #         if self.use_placeholder:
    #             content = self.name2placeholder[self.label2class[self.dataset[idx]['label']]] + f' {self.super_class_name}'
    #                     # Add the description to the content
    #             # content += f" in {self.dataset[idx]['description']}"
    #         else:
    #             content = self.label2class[self.dataset[idx]['label']] + f' {self.super_class_name}'
    #         # prompt =  random.choice(IMAGENET_TEMPLATES_SMALL).format(content)

    #         # Get the domain description
    #         domain_description = self.dataset[idx]['description']
            
    #         # Choose a template
    #         template = random.choice(IMAGENET_TEMPLATES_DIVE)
            
    #         # Find the position of the first space
    #         first_space_pos = template.find(' ')
            
    #         # Insert the domain description after the first word
    #         if first_space_pos != -1:
    #             # Split the template into the first word and the rest
    #             first_word = template[:first_space_pos]
    #             rest_of_template = template[first_space_pos:].lstrip()  # Remove leading spaces from the rest
    #             modified_template = f"{first_word} {domain_description} {rest_of_template}"
    #         else:
    #             # If there's no space in the template, just append the description
    #             modified_template = f"{template} {domain_description}"

    #         # Format the template with the content
    #         prompt = modified_template.format(content)
            
    #     else:
    #         prompt = random.choice(self.dataset[idx]['description'].strip().split('\n'))
    #     return prompt

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(name=self.label2class[self.get_label_by_idx(idx)])

    
if __name__== '__main__':
    ds = PersonHugDataset(examples_per_class=5)
    save_dir = '/data/zhicai/code/Diff-Mix/data/5shot'
    import tqdm
    for i in tqdm.tqdm(range(len(ds))):
        image = ds.get_image_by_idx(i)
        image.save(os.path.join(save_dir,f'{i}.png'))
    