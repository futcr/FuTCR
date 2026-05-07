#!/bin/bash

SCENARIO=100-50 
scenario_dir="${SCENARIO}_merged"



root_dir=MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/datasets
intg_output_dir="MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/splits_json"
custom_step_splits_dir="analyses/${scenario_dir}/custom_high_mid_low_cls_ov"
old_dir="MY_ROOT_FOLDER/research/continual_segmentation/method1/json/pan"   # Directory with old format files (val_100-5_step2_pan.json)
new_dir="$intg_output_dir/panoptic"  # Directory with new format files (test_100-5_step10_overlap100_panoptic.json)
output_dir="analyses/dataset_comparison_analysis"
scenarios=("${SCENARIO}") #("100-5", "100-10", "100-50", "50-10","50-20", "50-50") 

#100-5: 
# (100 100 75 75 50 50 25 25 0 75 50 25)
# (76  75  75 67 75 55 75 71 75 68 57 68)


# 100-10
# img_overlap_ratios=(100 100 75 75 75 50 50 50 25 25 25 0)  
# cls_overlap_ratios=(71 70 72 61 59 71 52 50 71 66 61 70) 

# 100-10 merged
img_overlap_ratios=(100 0) #(100 100 75 75 50 50 25 25 0 75 50 25)
cls_overlap_ratios=(75 75) #(76  75  75 67 75 55 75 71 75 68 57 68)

# Run dataset preparation workflow # --class_order
python complete_integration_workflow.py \
    --root_dir "$root_dir" \
    --output_dir "$intg_output_dir" \
    --scenarios "${scenarios[@]}" \
    --img_overlap_ratios "${img_overlap_ratios[@]}" \
    --cls_overlap_ratios "${cls_overlap_ratios[@]}" \
    --custom_step_splits_dir "${custom_step_splits_dir[@]}" \
    # --enable_cooccurrence && \ #no longer necessary
    # --randomize_class_order #


# python analyses/dataset_comparison_analyzer.py \
#     --old_dir "$old_dir" \
#     --new_dir "$new_dir" \
#     --output_dir "$output_dir" \
#     --scenarios "${scenarios[@]}" \
#     --img_overlap_ratios "${img_overlap_ratios[@]}" \
#     --cls_overlap_ratios "${cls_overlap_ratios[@]}" \
#     --standards old new
#     #--randomize_class_order 