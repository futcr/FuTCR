# method1 Batch Training System for Continual Learning
# Optimized for multiple overlap ratios with efficient GPU utilization

import os
import sys
from pathlib import Path

import os
import json
import copy
import tqdm
import numpy as np

#=======================method2===================================
def merge_prev_and_curr(prev_pan, prev_inst, curr_pan, curr_inst):
    merged_pan = copy.deepcopy(curr_pan)
    merged_inst = copy.deepcopy(curr_inst)
    for per_prev_img in prev_pan['images']:
        if per_prev_img['id'] not in [per_curr_img['id'] for per_curr_img in curr_pan['images']]:
            merged_pan['images'].append(per_prev_img)
    for per_prev_anno in prev_pan['annotations']:
        if per_prev_anno['image_id'] not in [per_curr_img['id'] for per_curr_img in curr_pan['images']]:
            merged_pan['annotations'].append(per_prev_anno)
        else:
            for per_curr_anno in merged_pan['annotations']:
                if per_curr_anno['image_id'] == per_prev_anno['image_id']:
                    per_curr_anno['segments_info'].extend(per_prev_anno['segments_info'])
                    break
    for per_prev_img in prev_inst['images']:
        if per_prev_img['id'] not in [per_curr_img['id'] for per_curr_img in curr_inst['images']]:
            merged_inst['images'].append(per_prev_img)
    for per_prev_anno in prev_inst['annotations']:
        merged_inst['annotations'].append(per_prev_anno)

    return merged_pan, merged_inst


def compute_global_nums(pan, global_segment_num=None):
    if global_segment_num is None:
        global_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in pan['categories']]}

    print("Computing global numbers...")
    for per_img_anno in tqdm.tqdm(pan['annotations']):
        segments_info = per_img_anno['segments_info']
        for seg in segments_info:
            global_segment_num[seg['category_id']] += 1

    target_segment_ratio = np.array(list(global_segment_num.values())) / np.sum(list(global_segment_num.values()))

    return target_segment_ratio, global_segment_num


def compute_stats(pan):
    segment_nums = {}

    print("Computing statistics...")
    for per_img_anno in tqdm.tqdm(pan['annotations']):
        per_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in pan['categories']]}

        segments_info = per_img_anno['segments_info']
        for seg in segments_info:
            per_segment_num[seg['category_id']] += 1
        segment_nums[per_img_anno['image_id']] = per_segment_num

    return segment_nums


def greedy_selection(
        images_data,
        num_categories,
        num_selections,
        target_segment_ratio,
        current_segment_num=None
):
    selected_images = []
    if current_segment_num is None:
        current_segment_num = np.zeros(num_categories)

    for _ in tqdm.tqdm(range(num_selections)):
        best_img = None
        best_score = float('inf')

        for img, segment_num in images_data.items():
            if img in selected_images:
                continue

            new_segment_num = (current_segment_num + np.array(segment_num))
            new_segment_ratio = new_segment_num / np.sum(new_segment_num)

            segment_diff = np.sum(np.abs(new_segment_ratio - target_segment_ratio))

            if segment_diff < best_score:
                best_score = segment_diff
                best_img = img

        selected_images.append(best_img)
        current_segment_num = current_segment_num + np.array(images_data[best_img])

    return selected_images, current_segment_num



def prepare_memory_cps_standardized(split, step, img_ov=100, cls_ov=75, 
                                    splits_base_dir="MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/splits_json",
                                    output_base_dir="./splits_json_mem/memory_splits"):
    """
    Prepare memory splits for continual learning with standardized paths.
    
    Args:
        split: Scenario like '100-5', '100-10', etc.
        step: Current step number
        img_ov: Image overlap percentage
        cls_ov: Class overlap percentage
        splits_base_dir: Base directory containing your generated splits
        output_base_dir: Where to save memory selections
        
    Returns:
        Dict with paths to created memory files
    """
    
    num_selections = 300
    
    # Parse scenario
    if split == '100-10':
        base_cls = 100
        inc_cls = 10
        keep_ratios = [100 / 110, 110 / 120, 120 / 130, 130 / 140, 140 / 150]
    elif split == '100-50':
        base_cls = 100
        inc_cls = 50
        keep_ratios = [100 / 150]
    elif split == '100-5':
        base_cls = 100
        inc_cls = 5
        keep_ratios = [100 / 105, 105 / 110, 110 / 115, 115 / 120, 120 / 125,
                       125 / 130, 130 / 135, 135 / 140, 140 / 145, 145 / 150]
    elif split == '50-50':
        base_cls = 50
        inc_cls = 50
        keep_ratios = [50 / 100, 100 / 150]
    else:
        raise ValueError(f"Unknown split: {split}")
    
    keep_ratio = 1 if step == 1 else keep_ratios[step - 2]

    # *** UPDATED: Use standardized paths ***
    splits_dir = Path(splits_base_dir)
    overlap_suffix = f"img_ov{img_ov}_cls_ov{cls_ov}"
    scenario_suffix = f"{base_cls}-{inc_cls}_step{step}_{overlap_suffix}"
    
    # Input paths (your generated splits)
    pan_path = splits_dir / "panoptic" / f"train_{scenario_suffix}_panoptic.json"
    inst_path = splits_dir / "instance" / f"train_{scenario_suffix}_instance.json"
    
    # Output paths (memory selections)
    output_dir = Path(output_base_dir) / overlap_suffix / "panoptic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pan_path = output_dir / f"train_{scenario_suffix}_memory_panoptic.json"
    output_inst_path = output_dir / f"train_{scenario_suffix}_memory_instance.json"
    output_target_path = output_dir / f"train_{scenario_suffix}_memory_target.json"
    
    print(f"Processing step {step} with {overlap_suffix}")
    print(f"  Input pan: {pan_path}")
    print(f"  Output pan: {output_pan_path}")
    
    # Load current step data
    with open(pan_path, 'r') as f:
        pan = json.load(f)
    with open(inst_path, 'r') as f:
        inst = json.load(f)

    if step > 1:
        # *** UPDATED: Load previous memory ***
        prev_scenario_suffix = f"{base_cls}-{inc_cls}_step{step - 1}_{overlap_suffix}"
        prev_pan_path = output_dir / f"train_{prev_scenario_suffix}_memory_panoptic.json"
        prev_inst_path = output_dir / f"train_{prev_scenario_suffix}_memory_instance.json"
        prev_target_path = output_dir / f"train_{prev_scenario_suffix}_memory_target.json"
        
        if not prev_pan_path.exists():
            raise FileNotFoundError(f"Previous memory not found: {prev_pan_path}\n"
                                   f"Please run memory selection for step {step-1} first")
        
        with open(prev_pan_path, 'r') as f:
            prev_pan = json.load(f)
        with open(prev_inst_path, 'r') as f:
            prev_inst = json.load(f)
        with open(prev_target_path, 'r') as f:
            prev_target = json.load(f)

        merged_pan, merged_inst = merge_prev_and_curr(prev_pan, prev_inst, pan, inst)

        prev_global_segment_nums = prev_target['global_segment_num']
        global_segment_num = \
            {int(cat_id): prev_global_segment_nums[cat_id] for cat_id in prev_global_segment_nums.keys()}

        target_segment_ratio, global_segment_num = compute_global_nums(pan, global_segment_num)
        segment_nums = compute_stats(merged_pan)

        prev_image_ids = [img['id'] for img in prev_pan['images']]
        prev_images_data = {}
        for img_id in prev_image_ids:
            prev_images_data[img_id] = list(segment_nums[img_id].values())
        
        print(f"  Selecting {int(num_selections * keep_ratio)} previous images...")
        selected_prev_images, current_segment_num = greedy_selection(
            images_data=prev_images_data,
            num_categories=target_segment_ratio.shape[0],
            num_selections=int(num_selections * keep_ratio),
            target_segment_ratio=target_segment_ratio,
        )

        images_data = {}
        for img_id in list(set(segment_nums.keys()) - set(prev_image_ids)):
            images_data[img_id] = list(segment_nums[img_id].values())
        
        print(f"  Selecting {num_selections - int(num_selections * keep_ratio)} current images...")
        selected_curr_images, current_segment_num = greedy_selection(
            images_data=images_data,
            num_categories=target_segment_ratio.shape[0],
            num_selections=num_selections - int(num_selections * keep_ratio),
            target_segment_ratio=target_segment_ratio,
            current_segment_num=current_segment_num,
        )
        selected_images = selected_prev_images + selected_curr_images

        pan = merged_pan
        inst = merged_inst

    else:
        target_segment_ratio, global_segment_num = compute_global_nums(pan)
        segment_nums = compute_stats(pan)

        images_data = {}
        for img_id in segment_nums.keys():
            images_data[img_id] = list(segment_nums[img_id].values())
        
        print(f"  Selecting {num_selections} images...")
        selected_images, current_segment_num = \
            greedy_selection(
                images_data=images_data,
                num_categories=target_segment_ratio.shape[0],
                num_selections=num_selections,
                target_segment_ratio=target_segment_ratio,
            )

    # Create memory splits with selected images
    new_pan = {
        'images': [],
        'annotations': [],
        'categories': pan['categories']
    }
    new_inst = {
        'images': [],
        'annotations': [],
        'categories': inst['categories']
    }

    for img in pan['images']:
        if img['id'] in selected_images:
            new_pan['images'].append(img)
    for ann in pan['annotations']:
        if ann['image_id'] in selected_images:
            new_pan['annotations'].append(ann)

    for img in inst['images']:
        if img['id'] in selected_images:
            new_inst['images'].append(img)
    for ann in inst['annotations']:
        if ann['image_id'] in selected_images:
            new_inst['annotations'].append(ann)

    # Save memory splits
    with open(output_pan_path, 'w') as f:
        json.dump(new_pan, f)
    with open(output_inst_path, 'w') as f:
        json.dump(new_inst, f)

    stats = {
        'global_segment_num': global_segment_num,
    }
    with open(output_target_path, 'w') as f:
        json.dump(stats, f)
    
    print(f"✅ Memory splits saved:")
    print(f"  Selected {len(selected_images)} images")
    print(f"  Pan: {output_pan_path}")
    print(f"  Inst: {output_inst_path}")
    
    return {
        'pan_path': str(output_pan_path),
        'inst_path': str(output_inst_path),
        'target_path': str(output_target_path),
        'num_images': len(selected_images)
    }


# *** ADD: Batch processing for all overlap combinations ***
def prepare_all_memory_splits(split='100-5', max_step=11, full_split_name='100-5'):
    """
    Prepare memory splits for all overlap combinations and steps.
    """
    
    
    #100-5 or 100-10 merged
    overlap_combinations= {    
        '100-5' : list(zip([0, 25,50, 75, 100, 25, 25, 50, 50, 75, 75, 100], 
                            [75,71,55, 67, 75,  68, 75, 57, 75, 68, 75, 76])),
        
        # Your overlap combinations
       
        
        '100-10_merged' :list(zip([0, 25,50, 75, 100, 25, 25, 50, 50, 75, 75, 100], 
                                    [75,71,55, 67, 75,  68, 75, 57, 75, 68, 75, 76])),
        
         # 100-10 main
        '100-10' : list(zip ([0, 25, 50, 75 ,100 ,100 ,75 ,75 ,50, 50, 25, 25],
                            [70, 66 ,52 ,59 ,71 ,70 ,72 ,61 ,71 ,50, 71 ,61])), #100-5
    }
    
    for img_ov, cls_ov in overlap_combinations[full_split_name]:
        print(f"\n{'='*60}")
        print(f"Processing img_ov={img_ov}, cls_ov={cls_ov}")
        print(f"{'='*60}")
        
        for step in range(1, max_step + 1):
            try:
                prepare_memory_cps_standardized(
                    split=split,
                    step=step,
                    img_ov=img_ov,
                    cls_ov=cls_ov, output_base_dir=f"./splits_json_mem_{full_split_name}/memory_splits"
                )
            except FileNotFoundError as e:
                print(f"⚠️  Skipping step {step}: {e}")
                continue
            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                continue







#=======================method2===================================









def get_split_paths(base_cls, inc_cls, task, img_ov=100, cls_ov=75, splits_base_dir="MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/splits_json"):
    """
    Get standardized split paths for any method.
    
    Args:
        base_cls: Number of base classes (e.g., 100)
        inc_cls: Number of incremental classes (e.g., 5)
        task: Task number (e.g., 2, 3, 4...)
        img_ov
        cls_ov
        splits_base_dir: Base directory containing splits
        
    Returns:
        Dict with all necessary paths for training
    """
    scenario=f"{base_cls}-{inc_cls}"
    splits_dir = Path(splits_base_dir)
    datasets_dir = Path("MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/datasets")
    
    # Base paths
    image_root_train = datasets_dir / "ADEChallengeData2016/images/training"
    image_root_val = datasets_dir / "ADEChallengeData2016/images/validation"
    panotpic_root_train = datasets_dir / "ADEChallengeData2016/ade20k_panoptic_train"
    panotpic_root_val = datasets_dir / "ADEChallengeData2016/ade20k_panoptic_val"
    detectron_train = datasets_dir / "ADEChallengeData2016/annotations_detectron2/training"
    detectron_val = datasets_dir / "ADEChallengeData2016/annotations_detectron2/validation"
    
    # Split-specific paths
    scenario_suffix = f"{scenario}_step{task}_img_ov{img_ov}_cls_ov{cls_ov}"
    
    split_paths = {
        # Training splits
        "train_images": str(image_root_train),
        "train_panoptic_root": str(panotpic_root_train),
        "train_panoptic_json": str(splits_dir / "panoptic" / f"train_{scenario_suffix}_panoptic.json"),
        "train_instance_json": str(splits_dir / "instance" / f"train_{scenario_suffix}_instance.json"),
        "train_sem_seg_root": str(detectron_train),
        
        # Validation splits  
        "val_images": str(image_root_val),
        "val_panoptic_root": str(panotpic_root_val),
        "val_panoptic_json": str(splits_dir / "panoptic" / f"val_{scenario_suffix}_panoptic.json"),
        "val_instance_json": str(splits_dir / "instance" / f"val_{scenario_suffix}_instance.json"),
        "val_sem_seg_root": str(detectron_val),
        
        # Metadata
        "scenario": scenario,
        "task": task,
        "img_ov":img_ov,
        "cls_ov": cls_ov,
        "base_cls": base_cls,
        "inc_cls": inc_cls
    }
    
    return split_paths




def get_standardized_predefined_split(cfg, img_ov=100, cls_ov=75):
    """
    Generate predefined split using our standardized paths.
    
    Args:
        cfg: Detectron2 config object
        img_ov
        cls_ov
        
    Returns:
        Dict with train/val split definitions for detectron2 registration
    """
    
    # Extract config parameters
    base_cls = cfg.CONT.BASE_CLS
    inc_cls = cfg.CONT.INC_CLS  
    task = cfg.CONT.TASK
    
    # Get standardized paths
    paths = get_split_paths(base_cls, inc_cls, task, img_ov, cls_ov)
    
    # Create predefined split structure matching method1 format
    predefined_split = {
        "current_ade20k_panoptic_train": (
            paths["train_images"],                    # Image root
            paths["train_panoptic_root"],            # Panoptic root 
            paths["train_panoptic_json"],            # Panoptic JSON
            paths["train_sem_seg_root"],             # Semantic seg root
            paths["train_instance_json"],            # Instance JSON
        ),
        "current_ade20k_panoptic_val": (
            paths["val_images"],                     # Image root
            paths["val_panoptic_root"],              # Panoptic root
            paths["val_panoptic_json"],              # Panoptic JSON  
            paths["val_sem_seg_root"],               # Semantic seg root
            paths["val_instance_json"],              # Instance JSON
        ),
    }
    
    return predefined_split


# Add this new function to split_paths_resolver.py

# Add to split_paths_resolver.py

def get_standardized_predefined_split_memory(cfg, img_ov=100, cls_ov=75 
                                            ):
    """
    Generate memory split using standardized memory selections.
    
    Args:
        cfg: Detectron2 config object
        img_ov: Image overlap percentage
        cls_ov: Class overlap percentage
        memory_base_dir: Base directory containing memory selections
        
    Returns:
        Dict with memory split definition for detectron2 registration
    """
    
    base_cls = cfg.CONT.BASE_CLS
    inc_cls = cfg.CONT.INC_CLS  
    task = cfg.CONT.TASK
    
    if task <= 1:
        raise ValueError("Memory split only available for task > 1")
    
    # Memory paths point to previous task's memory selection
    scenario = f"{base_cls}-{inc_cls}"
    overlap_suffix = f"img_ov{img_ov}_cls_ov{cls_ov}"
    scenario_suffix = f"{scenario}_step{task - 1}_{overlap_suffix}"
    memory_base_dir=f"MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/splits_json_mem_{scenario}/memory_splits" if int(inc_cls)> 5 else f"MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/splits_json_mem/memory_splits"
    memory_dir = Path(memory_base_dir) / overlap_suffix / "panoptic"
    datasets_dir = Path("MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/datasets")
    
    image_root_train = datasets_dir / "ADEChallengeData2016/images/training"
    panoptic_root_train = datasets_dir / "ADEChallengeData2016/ade20k_panoptic_train"
    detectron_train = datasets_dir / "ADEChallengeData2016/annotations_detectron2/training"
    
    predefined_split_memory = {
        "memory_ade20k_panoptic_train": (
            str(image_root_train),
            str(panoptic_root_train),
            str(memory_dir / f"train_{scenario_suffix}_memory_panoptic.json"),
            str(detectron_train),
            str(memory_dir / f"train_{scenario_suffix}_memory_instance.json"),
        ),
    }
    
    return predefined_split_memory



# if __name__ == "__main__":
#     # Test the function
#     paths = get_split_paths(100, 5, 2, 100, 75)
#     for key, value in paths.items():
#         print(f"{key}: {value}")
        
        
if __name__ == "__main__":
    print('main')
   
    # Or process all combinations UNCOMMENT THIS TO RUN MEMORY SPLITS
    # prepare_all_memory_splits(split='100-10', max_step=6,  full_split_name='100-10_merged')