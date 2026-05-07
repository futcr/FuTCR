# Standardized Continual Learning Dataset Preparation for ADE20K
# Compatible with: method1, method2, method3, method4, CoMFormer, PLOP
# Author: 
# Purpose: Unified dataset splitting with configurable overlap-disjoint ratios

import sys
import csv
import pandas as pd
import json
import os
import numpy as np
import random
from collections import OrderedDict, defaultdict
import argparse
from typing import List, Tuple, Dict, Union, Optional,Set, Any
import shutil
from pathlib import Path
from tools.advanced_swap_resplit_procedures import AdvancedSwapOverlaps, AdvancedResplitter
import matplotlib.pyplot as plt




random_seed=42
random.seed(random_seed)
np.random.seed(random_seed)
plot_colors = ['blue', 'green', 'red', 'orange', 'purple',   'black']

CUSTOM_OV_SPLITS_DIR='custom_overlap_splits_latest'

class StandardizedDatasetPreparator:
    """
    Standardized dataset preparation for continual learning in panoptic segmentation.
    Supports all major methods with configurable overlap-disjoint ratios.
    """
    
    #checked
    def __init__(self, 
                 threshold_p=100, threshold_h=50, enable_balancing=True,
                 root_dir: str = "datasets",
                 output_dir: str = "standardized_continual_splits",
                 custom_overlap_splits_dir:str = f"analyses/{CUSTOM_OV_SPLITS_DIR}",
                 img_overlap_ratio: int = 100,
                 cls_overlap_ratio: int = 100,
                 class_order: List[int] = list(range(0,150)),
                 randomize_class_order: bool = False,
                 random_seed: int = 42):
        """
        Initialize the dataset preparator.
        
        Args:
            root_dir: Root directory containing ADE20K dataset
            output_dir: Output directory for standardized splits  
            random_seed: Random seed for reproducibility
        """
        self.threshold_p = threshold_p
        self.threshold_h = threshold_h  
        self.enable_balancing = enable_balancing

        self.root_dir = root_dir
        self.output_dir = output_dir
        self.img_overlap_ratio = img_overlap_ratio
        self.cls_overlap_ratio = cls_overlap_ratio
        self.random_seed = random_seed
        
        # Standard paths following detectron2 format
        self.train_json_pan = os.path.join(root_dir, "ADEChallengeData2016/ade20k_panoptic_train.json")
        self.val_json_pan = os.path.join(root_dir, "ADEChallengeData2016/ade20k_panoptic_val.json")
        self.train_json_inst = os.path.join(root_dir, "ADEChallengeData2016/ade20k_instance_train.json")
        self.val_json_inst = os.path.join(root_dir, "ADEChallengeData2016/ade20k_instance_val.json")
        
        # Output directories for different formats (maintaining compatibility)
        self.pan_dir = os.path.join(output_dir, "panoptic")
        self.inst_dir = os.path.join(output_dir, "instance") 
        self.sem_dir = os.path.join(output_dir, "semantic")
        self.splits_dir = os.path.join(output_dir, "splits_metadata")
        self.custom_step_splits_dir = os.path.join(custom_overlap_splits_dir)
        
        # Create output directories
        for dir_path in [self.pan_dir, self.inst_dir, self.sem_dir, self.splits_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.categories_map = {}  
        self.class_order = class_order
        self.randomize_class_order = randomize_class_order
        
        # ADE20K class information (150 classes total)
        # First 100 are 'thing' classes, last 50 are 'stuff' classes
        self.total_classes = 150
        self.thing_classes = []  
        self.stuff_classes = []  
        
        # Instance-only class IDs (from method2/method3 analysis)
        # classes that appear as multiple distint instances and tracked individually regardless of thing or stuff
        self.instance_class_ids = [7, 8, 10, 12, 14, 15, 18, 19, 20, 22, 23, 24, 27, 30, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44,
                                  45, 47, 49, 50, 53, 55, 56, 57, 58, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 82,
                                  83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 97, 98, 102, 103, 104, 107, 108, 110, 111, 112, 115, 116, 118,
                                  119, 120, 121, 123, 124, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144,
                                  146, 147, 148, 149]
        
    #checked
    def load_dataset_info(self) -> Tuple[Dict, Dict]:
        """Load and analyze original ADE20K dataset."""
        print("Loading dataset information...")
        
        with open(self.train_json_pan, 'r') as f:
            train_data = json.load(f)
        with open(self.val_json_pan, 'r') as f:
            val_data = json.load(f)
            
            
        self.thing_classes, self.stuff_classes = self.load_categories_thing_stuff(
            train_data['categories'] 
        )
        
        
        
        # Analyze class distributions
        class_segment_counts = defaultdict(int)
        class_image_counts = defaultdict(int)
        
        for annotation in train_data['annotations']:
            image_classes = set()
            for segment in annotation['segments_info']:
                class_id = segment['category_id']
                class_segment_counts[class_id] += 1
                image_classes.add(class_id)
            for class_id in image_classes:
                class_image_counts[class_id] += 1
        
        print(f"Total training images: {len(train_data['images'])}")
        print(f"Total validation images: {len(val_data['images'])}")
        print(f"Classes found: {len(class_segment_counts)}")
        
        return train_data, val_data

    
    def load_step_data(self,step: int,
                    img_ov_pct: int,
                    cls_ov_pct: int,
                    full_train_images: List[Dict],
                    full_train_annotations: List[Dict],
                    base_output_dir: str,
                    scenario: str = "100-5") -> Tuple[List[Dict], List[Dict]]:
        """
        Load and filter actual image and annotation objects for a specific step.
        
        Args:
            step: Step number to load
            img_ov_pct: Image overlap percentage (e.g., 75 for 75%)
            cls_ov_pct: Class overlap percentage (e.g., 64 for 64%)
            full_train_images: Full list of image dictionaries
            full_train_annotations: Full list of annotation dictionaries
            base_output_dir: Base directory where splits are saved
            scenario: Scenario identifier
            
        Returns:
            Tuple of (filtered_images, filtered_annotations)
        """
        
        # Construct directory path
        dir_name = f"img_ov{img_ov_pct}_cls_ov{cls_ov_pct}"
        load_dir = Path(base_output_dir) / scenario / dir_name
        
        if not load_dir.exists():
            raise ValueError(f"Split directory not found: {load_dir}")
        
        # Load step image IDs
        step_data_path = load_dir / "step_image_ids.json"
        with open(step_data_path, 'r') as f:
            step_data = json.load(f)
        
        if str(step) not in step_data:
            raise ValueError(f"Step {step} not found in saved data. Available steps: {list(step_data.keys())}")
        
        img_ids = step_data[str(step)]['image_ids']
        img_id_set = set(img_ids)
        
        print(f"Loading step {step} from {dir_name}...")
        print(f"  Looking for {len(img_ids)} images")
        
        # Create lookup dictionaries for efficient filtering
        # Handle both dict and string image IDs
        img_lookup = {}
        for img in full_train_images:
            if isinstance(img, dict):
                img_id = img.get('id', img.get('file_name'))
            else:
                img_id = img
            img_lookup[img_id] = img
        
        ann_lookup = {}
        for ann in full_train_annotations:
            if isinstance(ann, dict):
                ann_id = ann.get('image_id', ann.get('id'))
            else:
                ann_id = ann
            ann_lookup[ann_id] = ann
        
        # Filter images and annotations
        filtered_images = []
        filtered_annotations = []
        
        for img_id in img_ids:
            if img_id in img_lookup:
                filtered_images.append(img_lookup[img_id])
                
                # Get corresponding annotation
                if img_id in ann_lookup:
                    filtered_annotations.append(ann_lookup[img_id])
                else:
                    print(f"    Warning: No annotation found for image {img_id}")
            else:
                print(f"    Warning: Image {img_id} not found in full dataset")
        
        print(f"    Loaded {len(filtered_images)} images and {len(filtered_annotations)} annotations")
        
        return filtered_images, filtered_annotations


    #checked
    def load_categories_thing_stuff(self, categories: List[Dict]) -> Tuple[List[int], List[int]]:
        """Load thing and stuff classes from categories metadata using isthing field."""
        catetories_map = {}
        thing_classes = []
        stuff_classes = []
        
        for category in categories:
            self.categories_map[category['id']] = dict(
                id =category['id'],
                name=category['name'],
                isthing=category['isthing']
                
                
                # 'id': idx,
                # 'name': name,
                # 'ratio': ratio,
                # 'train_count': train_count,
                # 'val_count': val_count
            )
            
            if category.get('isthing', 0) == 1:  # isthing = 1 means thing class
                thing_classes.append(category['id'])
            else:  # isthing = 0 means stuff class
                stuff_classes.append(category['id'])
                
        thing_classes = [c for c  in sorted(thing_classes)]
        stuff_classes = [c for c  in sorted(stuff_classes)]
        
        return   thing_classes, stuff_classes
    
    def log_to_csv(self,data_dict: Dict[str, Union[int, float, str]], filename: str = 'training_log.csv'):
        """
        Log data to a CSV file with specified columns.
        
        Parameters:
        data_dict (dict): Dictionary containing the data to log with keys:
                        'weight', 'task', 'num_ext_class', 'max_num_ext_classes', 
                        'num_task_classes', 'num_prev', 'num_future'
        filename (str): Name of the CSV file to write to
        """
        # Define the expected columns
        
        
        
        # Check if file exists
        file_exists = os.path.isfile(filename)
        
        # Open file in append mode
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(data_dict)
    
        print(f"Data logged to {filename}")
        

    #checked
    def create_randomizable_class_order(self, scenario: str, randomize=False) -> List[int]:
        """
        Create randomizable class ordering for different scenarios.
        
        Args:
            scenario: Format like "100-10", "50-50", etc.
            
        Returns:
            List of class IDs in task order
        """
        base_classes, inc_classes = map(int, scenario.split('-'))
        
        # Use all 150 classes, but randomize the order
        all_classes = self.class_order #list(range(self.total_classes))
        
        if randomize:
            random.shuffle(all_classes)
            
        
        # Ensure we have enough classes for the scenario
        total_needed = base_classes + inc_classes * 10  # Assume max 10 additional tasks
        if total_needed > self.total_classes:
            print(f"Warning: Scenario {scenario} needs {total_needed} classes, but only {self.total_classes} available")
            
        return all_classes[:total_needed] if total_needed <= self.total_classes else all_classes
    
    #checked
    def create_overlap_disjoint_split(self, 
                                    images: List,
                                    annotations: List, 
                                    task_classes: List[int],
                                    all_seen_classes: List[int], 
                                    use_probabilistic_control: bool = True, 
                                    task=None) -> Tuple[List, List]:
        """
        Create train/val splits with configurable overlap-disjoint ratio.
        
        Args:
            images: List of image info
            annotations: List of annotation info
            task_classes: Classes for current task
            all_seen_classes: All classes seen so far
            
        Returns:
            Tuple of (filtered_images, filtered_annotations)
        """
        
        USE_ALREADY_OPTIMIZED_SET = True 
        filtered_images = []
        filtered_annotations = []
        
        
        
        for image, annotation in zip(images, annotations):
            segments_info = annotation['segments_info']
            
            # Find all classes in this image
            image_classes = set([seg['category_id'] for seg in segments_info])
            
            # Determine if image should be included based on overlap ratio
            img_task_classes = image_classes.intersection(set(task_classes))
            img_future_classes = image_classes.difference(set(all_seen_classes))
            img_prev_classes = image_classes.intersection(set(all_seen_classes).difference(set(task_classes)))
            
            num_task = len(img_task_classes)
            num_prev = len(img_prev_classes)
            num_future = len(img_future_classes)
            
            has_task_classes = bool(img_task_classes)
            has_future_classes = bool(img_future_classes)
            has_prev_classes = bool(img_prev_classes)
            
            
            # Simple mode: just check for task class intersection
            if not use_probabilistic_control:
                if task == 1:
                    selection_criteria = has_task_classes #and has_future_classes #  (has_future_classes if random.random() <= 0.5 else not has_future_classes)  #not has_future_classes 
                else:
                    selection_criteria = has_task_classes  
                    
                if selection_criteria:
                    filtered_images.append(image)
                    # no filtering
                    filtered_annotations.append(annotation)
                continue
            
            
            
            
            
            # probabilistic inclusion based on overlap ratio  
            if has_task_classes:
                include_image =True
            else:
                include_image = False  
                
            
            if include_image:
                imgs =[image]
                anns = [annotation]
                
                
                updated_image_arr, updated_ann_arr = self.filter_annotations_to_task_classes(imgs, anns, task_classes)
                
                if updated_image_arr:
                    filtered_images.append(updated_image_arr[0])
                    
                if updated_ann_arr:
                    filtered_annotations.append(updated_ann_arr[0])
                
    
        
        return filtered_images, filtered_annotations
    
    #checked
    def create_standardized_test_split(self, val_data: Dict) -> Tuple[Dict, Dict]:
        """
        Create standardized validation (1k) and test (1k) splits from original validation set (2k).
        This ensures all methods use the same evaluation protocol.
        """
        print("Creating standardized val/test splits from original validation set...")
        
        images = val_data['images'].copy()
        annotations = val_data['annotations'].copy()
        
        # Shuffle with fixed seed for reproducibility
        combined = list(zip(images, annotations))
        random.shuffle(combined)
        images, annotations = zip(*combined)
        
        # Split into val (1k) and test (1k)
        mid_point = len(images) // 2
        
        val_split = {
            'images': list(images[:mid_point]),
            'annotations': list(annotations[:mid_point]), 
            'categories': val_data['categories']
        }
        
        test_split = {
            'images': list(images[mid_point:]),
            'annotations': list(annotations[mid_point:]),
            'categories': val_data['categories']
        }
        
        print(f"Val split: {len(val_split['images'])} images")
        print(f"Test split: {len(test_split['images'])} images")
        
        return val_split, test_split
    
    #checked
    def generate_scenario_splits(self, 
                            scenario: str,
                            train_data: Dict,
                            val_split: Dict,
                            test_split: Dict,
                            
                            ) -> None:
        """
        Generate all splits for a given scenario (e.g., "100-10").
        """
        base_classes, inc_classes = map(int, scenario.split('-'))
        
        if self.randomize_class_order :
            class_order = self.create_randomizable_class_order(scenario)
        else:
            class_order = self.class_order
        
        # Calculate number of tasks
        remaining_classes = len(class_order) - base_classes
        num_inc_tasks = (remaining_classes + inc_classes - 1) // inc_classes  # Ceiling division
        total_tasks = 1 + num_inc_tasks
        
        print(f"\nGenerating {scenario} scenario with {total_tasks} tasks...")
        print(f"Class order (first 20): {class_order[:20]}...")
        
        # Generate splits for each task
        for task in range(1, total_tasks + 1):
            if task == 1:
                # Base task
                task_classes = class_order[:base_classes]
                all_seen_classes = task_classes.copy()
            else:
                # Incremental task
                start_idx = base_classes + (task - 2) * inc_classes
                end_idx = min(start_idx + inc_classes, len(class_order))
                task_classes = class_order[start_idx:end_idx]
                all_seen_classes = class_order[:end_idx]
            
            print(f"Task {task}: Classes {task_classes[:5]}{'...' if len(task_classes) > 5 else ''} "
                f"({len(task_classes)} classes)")
            
            
            custom_overlap_step_imgs, custom_overlap_step_ann = self.load_step_data(
                task,
                self.img_overlap_ratio,
                self.cls_overlap_ratio,
                train_data['images'],
                train_data['annotations'],
                self.custom_step_splits_dir,
                scenario
            )
            
            if custom_overlap_step_imgs and custom_overlap_step_ann:
                main_train_imgs, main_train_anns = custom_overlap_step_imgs, custom_overlap_step_ann
            else:
                main_train_imgs, main_train_anns = train_data['images'], train_data['annotations']
            
            # Generate training split
            train_images, train_annotations = self.create_overlap_disjoint_split(
                main_train_imgs, main_train_anns,
                task_classes, all_seen_classes, task=task
            )
            print(f"FOR TASK {task}:before: imgs {len(main_train_imgs)} anns{len(main_train_anns)} \nAfter:\nimgs {len(train_images)} anns{len(train_annotations)}")
            
            # Generate validation split (always uses all seen classes)
            val_images, val_annotations = self.filter_by_classes(
                val_split['images'], val_split['annotations'], all_seen_classes
            )
            
            # Generate test split (always uses all seen classes)  
            test_images, test_annotations = self.filter_by_classes(
                test_split['images'], test_split['annotations'], all_seen_classes
            )
            
            # Save panoptic splits
            self.save_split_files(
                scenario, task, "panoptic",
                train_images, train_annotations, train_data['categories'],
                val_images, val_annotations, val_split['categories'],
                test_images, test_annotations, test_split['categories']
            )
            
            # Generate and save instance splits
            self.generate_instance_splits(
                scenario, task, task_classes, all_seen_classes,
                train_images, val_images, test_images
            )
        
        # Save metadata
        self.save_scenario_metadata(scenario, class_order, total_tasks)
    
    #checked
    def filter_annotations_to_task_classes(self,images, annotations, target_task_classes):
        """Filter annotations to keep only segments from target_task_classes."""
        filtered_images = []
        filtered_annotations = []
        
        for img, ann in zip(images, annotations):
            # Filter segments to keep only current task classes
            filtered_segments = []
            original_segments_catgs = set()
            
            for segment in ann['segments_info']:
                original_segments_catgs.add(segment['category_id'])
                
                # Keep only segments belonging to current task classes
                if segment['category_id'] in target_task_classes:
                    filtered_segments.append(segment)
            
            # Only include image if it has segments from current task
            if filtered_segments:
                # Create filtered annotation
                annotation_copy = ann.copy()
                annotation_copy['segments_info'] = filtered_segments
                filtered_annotations.append(annotation_copy)
                
                # Update image with category metadata
                updated_image = {**img, 'category_ids': list(original_segments_catgs)}
                filtered_images.append(updated_image)
        
        return filtered_images, filtered_annotations
    
    
    #checked
    def filter_by_classes(self, images: List, annotations: List, class_ids: List[int]) -> Tuple[List, List]:
        """Filter images and annotations to include only specified classes."""
        filtered_images = []
        filtered_annotations = []
        
        
        for image, annotation in zip(images, annotations):
            filtered_segments = []
            
            original_segments_catgs = set()
            updated_image = image
            
            for segment in annotation['segments_info']:
                original_segments_catgs.add(segment['category_id'])
                if segment['category_id'] in class_ids:
                    filtered_segments.append(segment)
            
            if filtered_segments:
                annotation_copy = annotation.copy()
                annotation_copy['segments_info'] = filtered_segments
                filtered_annotations.append(annotation_copy)
                
                filtered_images.append({**updated_image, 'category_ids': list(original_segments_catgs), })
        
        return filtered_images, filtered_annotations
    
    #checked
    def generate_instance_splits(self, 
                            scenario: str, 
                            task: int,
                            task_classes: List[int],
                            all_seen_classes: List[int],
                            train_images: List,
                            val_images: List, 
                            test_images: List) -> None:
        """Generate instance segmentation splits."""
        
        # Filter instance classes
        inst_task_classes = [cid for cid in task_classes if cid in self.instance_class_ids]
        inst_all_classes = [cid for cid in all_seen_classes if cid in self.instance_class_ids]
        
        if not inst_task_classes:
            print(f"No instance classes for task {task}, skipping instance splits")
            return
        
        # TODO: CHECK THE JSON FILE FOR BAD VALUES IN ANNOTATIONS EG.COUNT
        # Load instance data
        with open(self.train_json_inst, 'r') as f:
            train_inst_data = json.load(f)
        with open(self.val_json_inst, 'r') as f:
            val_inst_data = json.load(f)
        
        # Filter instance annotations
        train_image_ids = set([img['id'] for img in train_images])
        val_image_ids = set([img['id'] for img in val_images])
        test_image_ids = set([img['id'] for img in test_images])
        
        # Training instance split
        train_inst_anns = [ann for ann in train_inst_data['annotations'] 
                        if ann['image_id'] in train_image_ids and ann['category_id'] in inst_all_classes]
        train_inst_imgs = [img for img in train_inst_data['images'] if img['id'] in train_image_ids]
        
        # Validation instance split (using original val data)
        val_inst_anns = [ann for ann in val_inst_data['annotations']
                        if ann['image_id'] in val_image_ids and ann['category_id'] in inst_all_classes]
        val_inst_imgs = [img for img in val_inst_data['images'] if img['id'] in val_image_ids]
        
        # Test instance split (using original val data)
        test_inst_anns = [ann for ann in val_inst_data['annotations']
                        if ann['image_id'] in test_image_ids and ann['category_id'] in inst_all_classes]
        test_inst_imgs = [img for img in val_inst_data['images'] if img['id'] in test_image_ids]
        
        # Save instance splits
        self.save_instance_files(
            scenario, task,
            train_inst_imgs, train_inst_anns, train_inst_data['categories'],
            val_inst_imgs, val_inst_anns, val_inst_data['categories'],
            test_inst_imgs, test_inst_anns, val_inst_data['categories']
        )
    
    #checked
    def save_split_files(self,
                        scenario: str,
                        task: int, 
                        split_type: str,
                        train_images: List, train_annotations: List, train_categories: List,
                        val_images: List, val_annotations: List, val_categories: List,
                        test_images: List, test_annotations: List, test_categories: List) -> None:
        """Save split files in standardized format."""
        
        overlap_suffix = f"_img_ov{self.img_overlap_ratio}_cls_ov{self.cls_overlap_ratio}"
        
        def set_default(o):
            if isinstance(o, set):
                return list(o)
            raise TypeError(f"Type {type(o)} not serializable")
        
        
        # Training split
        train_split = {
            'images': list(train_images),
            'annotations': list(train_annotations), 
            'categories': list(train_categories)
        }
        train_path = os.path.join(self.pan_dir, f"train_{scenario}_step{task}{overlap_suffix}_{split_type}.json")
        with open(train_path, 'w') as f:
            json.dump(train_split, f, default=set_default, indent=2)
        # print(f"Saved training split: {train_path} ({len(train_images)} images)")
        
        # Validation split  
        val_split = {
            'images': list(val_images),
            'annotations': list(val_annotations),
            'categories': list(val_categories)
        }
        val_path = os.path.join(self.pan_dir, f"val_{scenario}_step{task}{overlap_suffix}_{split_type}.json") 
        with open(val_path, 'w') as f:
            json.dump(val_split, f, default=set_default, indent=2)
        # print(f"Saved validation split: {val_path} ({len(val_images)} images)")
        
        # Test split (new!)
        test_split = {
            'images': list(test_images),
            'annotations': list(test_annotations),
            'categories': list(test_categories)
        }
        test_path = os.path.join(self.pan_dir, f"test_{scenario}_step{task}{overlap_suffix}_{split_type}.json")
        with open(test_path, 'w') as f:
            json.dump(test_split, f, default=set_default, indent=2)
        # print(f"Saved test split: {test_path} ({len(test_images)} images)")
    
    #checked
    def save_instance_files(self, 
                        scenario: str,
                        task: int,
                        train_images: List, train_annotations: List, train_categories: List,
                        val_images: List, val_annotations: List, val_categories: List,
                        test_images: List, test_annotations: List, test_categories: List) -> None:
        """Save instance segmentation files."""
        
        overlap_suffix = f"_img_ov{self.img_overlap_ratio}_cls_ov{self.cls_overlap_ratio}"
        
        # Training instance split
        train_inst = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': train_categories
        }
        train_path = os.path.join(self.inst_dir, f"train_{scenario}_step{task}{overlap_suffix}_instance.json")
        with open(train_path, 'w') as f:
            json.dump(train_inst, f)
        
        # Validation instance split
        val_inst = {
            'images': val_images, 
            'annotations': val_annotations,
            'categories': val_categories
        }
        val_path = os.path.join(self.inst_dir, f"val_{scenario}_step{task}{overlap_suffix}_instance.json")
        with open(val_path, 'w') as f:
            json.dump(val_inst, f)
            
        # Test instance split
        test_inst = {
            'images': test_images,
            'annotations': test_annotations,
            'categories': test_categories
        }
        test_path = os.path.join(self.inst_dir, f"test_{scenario}_step{task}{overlap_suffix}_instance.json")
        with open(test_path, 'w') as f:
            json.dump(test_inst, f)
    
    #checked
    def get_category_name(self, category_id: int) -> str:
        """Get category name from objectInfo150.txt mapping or fallback to ID."""
        if category_id in self.categories_map:
            return self.categories_map[category_id]['name']
        else:
            return f"category_{category_id}"
    
    #checked
    def save_scenario_metadata(self, scenario: str, class_order: List[int], total_tasks: int) -> None:
        """Save scenario metadata for reproducibility and analysis."""
        overlap_suffix = f"_img_ov{self.img_overlap_ratio}_cls_ov{self.cls_overlap_ratio}"
        
        thing_classes_in_order = [cid for cid in class_order if cid in self.thing_classes]
        stuff_classes_in_order = [cid for cid in class_order if cid in self.stuff_classes]
        class_names_map ={}
        
        for cid in class_order:
            class_names_map[cid] = self.get_category_name(cid)
        
        
        metadata = {
            'scenario': scenario,
            'img_overlap_ratio': self.img_overlap_ratio,
            'cls_overlap_ratio': self.cls_overlap_ratio,
            'random_seed': self.random_seed,
            'total_tasks': total_tasks,
            'class_order': class_order,
            'total_classes': len(class_order),
            'thing_classes': thing_classes_in_order,  # Use actual isthing=1 classes
            'stuff_classes': stuff_classes_in_order,  # Use actual isthing=0 classes
            'total_thing_classes': len(thing_classes_in_order),
            'total_stuff_classes': len(stuff_classes_in_order),
            'instance_classes': [cid for cid in class_order if cid in self.instance_class_ids],
            'class_names_for_class_order': class_names_map,  # Add category names
            

        }
        
        metadata_path = os.path.join(self.splits_dir, f"{scenario}{overlap_suffix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved Scenario metadata: {metadata_path}")
    
    #checked
    def create_dataloader_config(self) -> None:
        """Create optimized dataloader configuration file."""
        
        dataloader_config = {
            "general": {
                "num_workers": 8,  # Optimal for most systems
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2
            },
            "training": {
                "batch_size": 16,  # Adjust based on GPU memory
                "shuffle": True,
                "drop_last": True
            },
            "validation": {
                "batch_size": 32,  # Can be larger for validation
                "shuffle": False,
                "drop_last": False
            },
            "test": {
                "batch_size": 32,
                "shuffle": False, 
                "drop_last": False
            }
        }
        
        config_path = os.path.join(self.output_dir, "dataloader_config.json")
        with open(config_path, 'w') as f:
            json.dump(dataloader_config, f, indent=2)
        print(f"Saved dataloader config: {config_path}")
    
    #checked
    def create_method_compatibility_configs(self) -> None:
        """Create configuration files for different methods."""
        
        # method1 compatibility
        method1_config = {
            "data_root": self.output_dir,
            "panoptic_path": "panoptic",
            "instance_path": "instance", 
            "file_format": "{split}_{scenario}_step{task}_overlap{overlap}_{type}.json"
        }
        
        # method2 compatibility  
        method2_config = {
            "json_root": self.output_dir,
            "pan_dir": "panoptic",
            "inst_dir": "instance",
            "file_format": "{split}_{scenario}_step{task}_overlap{overlap}_{type}.json"
        }
        
        # Save configs
        configs_dir = os.path.join(self.output_dir, "method_configs")
        os.makedirs(configs_dir, exist_ok=True)
        
        with open(os.path.join(configs_dir, "method1_config.json"), 'w') as f:
            json.dump(method1_config, f, indent=2)
            
        with open(os.path.join(configs_dir, "method2_config.json"), 'w') as f:
            json.dump(method2_config, f, indent=2)
        
        print(f"Saved method compatibility configs in {configs_dir}")
        
    def compute_global_cooccurrence_matrix(self, images, annotations, all_classes):
        pass
        # Compute symmetric O_D×D matrix for all classes
        # Implementation provided in cooccurrence_overlap_control.py
        
    
    def enhanced_swap_images_for_overlap_reduction(self, ): # ...other params
        # Use AdvancedSwapOverlaps instead of basic swapping logic
        advanced_swapper = AdvancedSwapOverlaps(priority=('test', 'base'))
        return advanced_swapper.swap(IC_imgs, IB_imgs, Itest_imgs, target_category, min_r)


def extract_weight_statistics(filename: str = 'training_log.csv') -> Tuple[float, float, float]:
    """
    Extract max, mean, and min values from the weight column of the CSV file.
    
    Parameters:
    filename (str): Name of the CSV file to read from
    
    Returns:
    tuple: (max_weight, mean_weight, min_weight)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filename)
        
        # Check if weight column exists
        if 'ext_max_ext_ratio' not in df.columns:
            raise ValueError("Weight column not found in the CSV file")
        
        # Extract statistics
        max_weight = df['ext_max_ext_ratio'].max()
        mean_weight = df['ext_max_ext_ratio'].mean()
        min_weight = df['ext_max_ext_ratio'].min()
        
        print(f"Weight Statistics:")
        print(f"Maximum: {max_weight}")
        print(f"Mean: {mean_weight:.4f}")
        print(f"Minimum: {min_weight}")
        
        return max_weight, mean_weight, min_weight
        
    except FileNotFoundError:
        print(f"File {filename} not found. Please ensure the file exists.")
        return None, None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None
    


def main():
    parser = argparse.ArgumentParser(description="Standardized Continual Learning Dataset Preparation")
    parser.add_argument("--root_dir", default="datasets", help="Root directory containing ADE20K")
    parser.add_argument("--output_dir", default="standardized_continual_splits", help="Output directory")
    parser.add_argument("--img_overlap_ratio", type=float, default=1.0, 
                       help="Overlap ratio (1.0=100%% overlap, 0.0=disjoint)")
    parser.add_argument("--scenarios", nargs='+', default=["100-10", "100-5",  "100-50", "50-50", "50-50", "50-50"],
                       help="Scenarios to generate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--create_multiple_overlaps", action='store_true',
                       help="Create multiple overlap ratios (100, 75, 50, 25, 0)")
    parser.add_argument("--class_order", nargs='+', type=int, default=list(range(0,150)),
                       help="Order of classes (e.g., --class_order 1.0 0.5 0.0)")
    parser.add_argument("--randomize_class_order", type=bool, default=False,
                       help="Whether to randomize the classes")

    args = parser.parse_args()
    
    overlap_ratios = [1.0, 0.75, 0.5, 0.25, 0.0] if args.create_multiple_overlaps else [args.img_overlap_ratio]
    cls_overlap_ratio = 1.0
    for img_overlap_ratio in img_overlap_ratios:
        
        # Initialize preparator
        preparator = StandardizedDatasetPreparator(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            img_overlap_ratio=img_overlap_ratio,
            cls_overlap_ratio=cls_overlap_ratio,
            class_order = args.class_order,
            random_seed=args.random_seed,
            randomize_class_order = args.randomize_class_order
        )
        
        # Load dataset
        train_data, val_data = preparator.load_dataset_info()
        
        # Create standardized val/test split
        val_split, test_split = preparator.create_standardized_test_split(val_data)
        
        # Generate scenarios
        for scenario in args.scenarios:
            # TODO: MODIFY CLASS ORDER AND CLASS ORDERING METHOD TO GET A PARTICULAR non-randomized CLASS ARRANGMENT
            preparator.generate_scenario_splits(scenario, train_data, val_split, test_split)
        

    
    print(f"\n{'='*80}")
    print("Dataset preparation completed!")
    print("Next steps:")
    print("1. Point your method's config to the standardized splits")
    print("2. Use the test splits for evaluation")  
    print("3. Compare results across different overlap ratios")
    print("4. Use the dataloader config for optimal performance")
    print(f"{'='*80}")


if __name__ == "__main__":
    extract_weight_statistics("MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/ordered_classes_image_composition.csv")
    # main()