#!/bin/bash
# t=${1:-11}
# python train_continual.py --dist-url auto \
#   --eval-only --num-gpus 4 \
#   --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
#   CONT.TASK ${t} CONT.VQ_NUMBER 0 \
#   OUTPUT_DIR ./output/ps/100-5_607/step${t} \


# OUTPUT_DIR=MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/methods/method1/output_cache/ps/img_ov100_cls_ov76/100-5/step1

# OUTPUT_DIR=MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/methods/method1/futcr_author2s_output/step1

#!/usr/bin/env bash

set -euo pipefail

# futcr_author1: list of top-level experiment dirs (relative to some root)
MAIN_DIRS=(
	"ps_FA_SEED2_L0.5_RCMT0.5_RCNP50_IPW0.5_KDD_True_100-50"
	"ps_FA_SEED2_L0.5_RCMT0.5_RCNP50_IPW0.5_KDD_False_100-50"
	"ps_Baseline_seed2_KDD_True_100-50"
	"ps_Baseline_seed2_KDD_False_100-50"
	# "ps_Baseline_seed2_KDD_True_100-10"
	# "ps_Baseline_seed2_KDD_False_100-10"
	# "ps_FA_SEED2_L0.5_IPW0.5_KDD_True_100-10"
	# "ps_FA_SEED2_L0.5_RCMT0.5_RCNP50_IPW0.5_KDD_False_100-10"
	# "ps_FA_SEED2_L0.5_RCMT0.5_RCNP50_IPW0.5_KDD_True_100-10"
	# "ps_Baseline_seed2"
	# "ps_FA_SEED2_L0.5_RCMT0.5_RCNP50_IPW0.5"
	# "ps_FA_SEED2_L0.5_RCMT0.5_RCNP50_ACNC20"
	# "ps_FA_SEED2_L0.5_IPW0.5_ACNC20"

	# "ps_FA_L0.5_RCMT0.5_RCNP48_IPW0.5_ACNC25"
	# "ps_FA_seed2_L0.5_RCMT0.5_RCNP50_IPW0.5_ACNC20"
	#   "ps_FA_seed2_L0.5_RCMT0.5_RCNP50_IPW0.5_ACNC20"
	#   "ps_FA_L1.0_RCMT0.5_RCNP48_IPW0.5_ACNC20"
	# "ps_FA_L0.5_RCMT0.5_RCNP25_IPW0.5_ACNC20"
	# "ps_FA_L1.5_RCMT0.5_RCNP48_IPW0.5_ACNC20"
	# "ps_FA_L0.1_RCMT0.4_RCNP70_IPW0.5_ACNC20"
	# "ps_FA_L0.5_RCMT0.5_RCNP48_IPW0.5_ACNC20"
	# "ps_future_aware_L1.0"
# "ps"
#   "ps_FA_L0.25_RCMT0.35_RCNP100_IPW0.5_ACNC15"
#   "ps_FA_L0.5_RCMT0.4_RCNP70_IPW0.5_ACNC20"
#   "ps_FA_L0.1_RCMT0.5_RCNP48_IPW0.5_ACNC20"
)

# futcr_author1: list of subdirs under each main dir
SUB_DIRS=(
   	# "img_ov100_cls_ov75"
	# "img_ov75_cls_ov67"
	#"img_ov50_cls_ov55"
	#"img_ov25_cls_ov71"
    "img_ov0_cls_ov75"
)

ROOT_OUTPUT="MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/methods/method1/output"
EVAL_ROOT="${ROOT_OUTPUT}/eval_only"

# Task id (default 11 if not passed)

SCENARIO="100-50"
t="${1:-2}"

APPEND_TEXT=$'=====================================================================\n=====================================================================\n======================= TEST SET=================================\n=====================================================================\n====================================================================='
APPEND_END_TEXT=$'===============================TEST ENDED!!!=========================='
for main in "${MAIN_DIRS[@]}"; do
	for sub in "${SUB_DIRS[@]}"; do
		model_dir="${main}/${sub}/${SCENARIO}/step${t}"

		# Original training output dir
		OUTPUT_DIR="${ROOT_OUTPUT}/${model_dir}"

		# Eval-only output dir
		EVAL_OUTPUT_DIR="${EVAL_ROOT}/${model_dir}"

		# mkdir -p "${EVAL_OUTPUT_DIR}"

		file_to_append="${OUTPUT_DIR}/log.txt"
		echo appending to $file_to_append
		# Append the separator block
		printf '%s\n' "${APPEND_TEXT}" >> "${file_to_append}"
		printf '%s\n' "${APPEND_TEXT}" >> "${file_to_append}"

		python train_continual_eval.py --dist-url auto \
		--eval-only --num-gpus 1 \
		--config-file configs/ade20k/panoptic-segmentation/${SCENARIO}.yaml \
		CONT.TASK "${t}" CONT.VQ_NUMBER 0 \
		OUTPUT_DIR "${OUTPUT_DIR}"
		printf '%s\n' "${APPEND_END_TEXT}" >> "${file_to_append}"
		printf '%s\n' "${model_dir}" >> "${file_to_append}"
		printf '%s\n' "${APPEND_END_TEXT}" >> "${file_to_append}"
	done
done



# model_dir=ps_FA_L0.25_RCMT0.35_RCNP100_IPW0.5_ACNC15/img_ov100_cls_ov75/100-5/step11
# OUTPUT_DIR="MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/methods/method1/output/$model_dir"
# OUTPUT_DIR=MY_ROOT_FOLDER/research/continual_segmentation/standard_framework/methods/method1/output/eval_only/$model_dir

# t=${1:-11}
# python train_continual_eval.py --dist-url auto \
#   --eval-only --num-gpus 1 \
#   --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
#   --save_dir ${SAVE_DIR} \
#   CONT.TASK ${t} CONT.VQ_NUMBER 0 \
#   OUTPUT_DIR "$OUTPUT_DIR" \