#!/bin/bash
# method1 Pan Continual Training Script - Handles Base and All Continual Steps
# Usage: bash pan_100-5_continual.sh <img_overlap_pct> <cls_overlap_pct> <base_cls> <inc_cls>

set -e  # Exit on any error



gpu_ids=$(python -c "import os;print(os.getenv('CUDA_VISIBLE_DEVICES'))")
echo "GPU IDS: $gpu_ids"
count=$(echo "$gpu_ids" | grep -o '[0-9]\+' | wc -l)
ngpus=$count
echo "NUMBER OF GPUS ABOUT TO BE USED: $ngpus"
nvidia-smi

# ================================================================
#                    PARSE ARGUMENTS
# ================================================================
IMG_OV_PCT=${1:-100}
CLS_OV_PCT=${2:-75}
BASE_CLS=${3:-100}
INC_CLS=${4:-5}
BATCH_TASK_IDX=${5:-0}

#--------FUTURE AWARE ARGUMENTS-------------
LOSS_WEIGHT=${6:-0.5}
REGION_CONTRAST_ENABLE=${7:-True}
MASK_THRESHOLD=${8:-0.5}
NUM_SAMPLED_PIXELS_PER_REGION=${9:-50}
IGNORE_REPULSION_ENABLE=${10:-True}
IGNORE_REPULSION_WEIGHT=${11:-0.5}
AUX_CLS_ENABLE=${12:-True}
AUX_CLS_NUM_CLUSTERS_MULTIPLIER=${13:-4}
EXP_SEED=${14:-2}



TASKNAME="ps_FA_SEED${EXP_SEED}"

TOTAL_CLASSES=150
SCENARIO="${BASE_CLS}-${INC_CLS}"
REMAINING_CLASSES=$((TOTAL_CLASSES - BASE_CLS))
NUM_TASKS=$(((REMAINING_CLASSES + INC_CLS - 1) / INC_CLS))

# Format directory name
OVERLAP_DIR="img_ov${IMG_OV_PCT}_cls_ov${CLS_OV_PCT}"


# ================================================================
#                    INCLUDE EXTRA HYPERPARAMETERS
# ================================================================
FUTURE_AWARE_ENABLE=True

AUX_CLS_NUM_CLUSTERS=$(((AUX_CLS_NUM_CLUSTERS_MULTIPLIER) * INC_CLS ))

OUTPUT_FLAG=""


extra_args=()
if [[ "$FUTURE_AWARE_ENABLE" == "True" ]]; then
    extra_args+=(CONT.FUTURE_AWARE.ENABLE True)

    if [[ "$LOSS_WEIGHT" != "" ]]; then
        extra_args+=(CONT.FUTURE_AWARE.LOSS_WEIGHT "$LOSS_WEIGHT")
        OUTPUT_FLAG+="_L${LOSS_WEIGHT}"
    fi

    if [[ "$REGION_CONTRAST_ENABLE" == "True" ]]; then
        extra_args+=(CONT.FUTURE_AWARE.REGION_CONTRAST_ENABLE True)
        extra_args+=(CONT.FUTURE_AWARE.MASK_THRESHOLD "$MASK_THRESHOLD")
        extra_args+=(CONT.FUTURE_AWARE.NUM_SAMPLED_PIXELS_PER_REGION "$NUM_SAMPLED_PIXELS_PER_REGION")
        OUTPUT_FLAG+=_RCMT${MASK_THRESHOLD}_RCNP${NUM_SAMPLED_PIXELS_PER_REGION}
    fi

    if [[ "$IGNORE_REPULSION_ENABLE" == "True" ]]; then
        extra_args+=(CONT.FUTURE_AWARE.IGNORE_REPULSION_ENABLE True)
        extra_args+=(CONT.FUTURE_AWARE.IGNORE_REPULSION_WEIGHT "$IGNORE_REPULSION_WEIGHT")
        OUTPUT_FLAG+=_IPW${IGNORE_REPULSION_WEIGHT}
    fi

    if [[ "$AUX_CLS_ENABLE" == "True" ]]; then
        extra_args+=(CONT.FUTURE_AWARE.AUX_CLS_ENABLE True)
        extra_args+=(CONT.FUTURE_AWARE.AUX_CLS_NUM_CLUSTERS "$AUX_CLS_NUM_CLUSTERS")
        OUTPUT_FLAG+=_ACNC${AUX_CLS_NUM_CLUSTERS}
    fi
fi




RESULT_DIR="./output/${TASKNAME}${OUTPUT_FLAG}/${OVERLAP_DIR}/${SCENARIO}"

echo "=========================================="
echo "method1 Complete Training Pipeline"
echo "Image Overlap: ${IMG_OV_PCT}%"
echo "Class Overlap: ${CLS_OV_PCT}%"
echo "Overlap Directory: ${OVERLAP_DIR}"
echo "Scenario: ${SCENARIO}"
echo "Total Tasks: $((NUM_TASKS + 1)) (1 base + ${NUM_TASKS} continual)"
echo "=========================================="


# ================================================================
#          IMAGE COUNT SELECTION & ITER FOR QUERY COLLECTION
# ================================================================

IMS_PER_BATCH=8

# method1-specific QUERY_MODE_ITERATIONS counts for query collection
BASE_IMG_COUNT=10177
BASE_QUERY_MODE_ITERATIONS=$(( (BASE_IMG_COUNT + IMS_PER_BATCH - 1) / IMS_PER_BATCH ))



if [ "$IMG_OV_PCT" -eq 0 ]; then
    STEP_IMG_COUNTS=(569 297 617 598 570 935 903 1469 617 945)
else
    STEP_IMG_COUNTS=(573 297 617 598 581 941 903 1475 617 953)
fi

STEP_QUERY_MODE_ITERATIONS=()

for i in "${!STEP_IMG_COUNTS[@]}"; do
    step_iter=$(( (STEP_IMG_COUNTS[$i] + IMS_PER_BATCH - 1) / IMS_PER_BATCH ))
    STEP_QUERY_MODE_ITERATIONS+=($step_iter)
done

echo "BASE_QUERY_MODE_ITERATIONS: ${BASE_QUERY_MODE_ITERATIONS} STEP_QUERY_MODE_ITERATIONS: ${STEP_QUERY_MODE_ITERATIONS[@]}"




# ================================================================
#                    BASE SETTINGS
# ================================================================


# Decay at 84% and 94% of training futcr_author1: adopted from https://huggingface.co/LightningNO1/method1/blob/main/ps_100base/step1/config.yaml
BASE_STEPS="(135000,150000)"  


BASE_MODEL_PATH="${RESULT_DIR}/step1"


# Check if base training is needed
if [[ -d "$BASE_MODEL_PATH" ]] && [[ -f "$BASE_MODEL_PATH/model_final.pth" ]]; then
    echo "Base model already exists at: $BASE_MODEL_PATH"
    TRAIN_BASE=false
else
    echo " Base model not found, will train base (step 1) first"
    TRAIN_BASE=true
fi


#===================== resume flag =================
if [[ -f "$BASE_MODEL_PATH/last_checkpoint" ]]; then
        echo "  Found existing checkpoint, resuming training..."
        RESUME_FLAG="--resume"
    else
        echo "Starting training from scratch..."
        RESUME_FLAG=""
    fi


# ================================================================
#                    BASE TRAINING (STEP 1)
# TODO: change max_iter for other scenarios? 
#   NB: added SOLVER.STEPS from hugging face config
# ================================================================
if [ "$TRAIN_BASE" = true ]; then
    echo ""
    echo "=========================================="
    echo "Training Base Model (Step 1)"
    echo "=========================================="
    
    mkdir -p "$BASE_MODEL_PATH"
    
#    # QUERY COLLECTION
    NCCL_P2P_LEVEL=LOC CUDA_LAUNCH_BLOCKING=1 python train_continual.py \
        --dist-url auto \
        --num-gpus 1 \
        --config-file configs/ade20k/panoptic-segmentation/${SCENARIO}.yaml \
        --img-overlap "${IMG_OV_PCT}" \
        --cls-overlap "${CLS_OV_PCT}" \
        $RESUME_FLAG \
        CONT.TASK 1 \
        "${extra_args[@]}" \
        SOLVER.BASE_LR 0.0 \
        SOLVER.MAX_ITER ${BASE_QUERY_MODE_ITERATIONS} \
        CONT.COLLECT_QUERY_MODE True \
        SEED $EXP_SEED \
        OUTPUT_DIR "$BASE_MODEL_PATH" \
        SOLVER.STEPS ${BASE_STEPS} \
        



   # MAIN BASE TRAINING
    NCCL_P2P_LEVEL=LOC CUDA_LAUNCH_BLOCKING=1 python train_continual.py \
        --dist-url auto \
        --num-gpus $ngpus \
        --config-file configs/ade20k/panoptic-segmentation/${SCENARIO}.yaml \
        --img-overlap "${IMG_OV_PCT}" \
        --cls-overlap "${CLS_OV_PCT}" \
        $RESUME_FLAG \
        CONT.TASK 1 \
        "${extra_args[@]}" \
        SOLVER.BASE_LR 0.0001 \
        SOLVER.MAX_ITER 160000 \
        CONT.COLLECT_QUERY_MODE False \
        SEED $EXP_SEED \
        OUTPUT_DIR "$BASE_MODEL_PATH" \
        SOLVER.STEPS ${BASE_STEPS} \
    
    if [ $? -eq 0 ]; then
        echo " Base training completed successfully!"
        echo "$(date): Completed base training" >> training_log.txt
    else
        echo " Base training failed!"
        exit 1
    fi
    
    echo ""

    TRAIN_BASE=false
fi










# ================================================================
#                    CONTINUAL TRAINING (STEPS 2+)
# ================================================================
echo "=========================================="
echo "Starting Continual Training (Steps 2-$((NUM_TASKS + 1)))"
echo "=========================================="


if [ "$TRAIN_BASE" = false ]; then
    # Train all continual tasks sequentially
    for task_id in $(seq 2 $((NUM_TASKS + 1))); do
        echo ""
        echo "  Training Task ${task_id} / $((NUM_TASKS + 1))"
        echo "$(date): Starting task ${task_id}"
        
        # Output directory for this task
        OUTPUT_DIR="${RESULT_DIR}/step${task_id}"
        mkdir -p "$OUTPUT_DIR"
        
        # Get STEP_QUERY_MODE_ITERATIONS count for query collection
        index=$((task_id - 2))
        
        if [ $index -ge 0 ] && [ $index -lt ${#STEP_QUERY_MODE_ITERATIONS[@]} ]; then
            iter=${STEP_QUERY_MODE_ITERATIONS[$index]}
            
            echo "  Phase 1: Collecting virtual queries (${iter} STEP_QUERY_MODE_ITERATIONS)..."
             #===================== resume flag =================

            # *** CHECK FOR EXISTING CHECKPOINT ***
            if [[ -f "$OUTPUT_DIR/last_checkpoint" ]]; then
                echo "  Found existing checkpoint for task ${task_id}, resuming..."
                RESUME_FLAG="--resume"
            else
                RESUME_FLAG=""
            fi

            # Collect virtual queries
            python train_continual.py \
                --dist-url auto \
                --num-gpus 1 \
                --config-file configs/ade20k/panoptic-segmentation/${SCENARIO}.yaml \
                --img-overlap "${IMG_OV_PCT}" \
                --cls-overlap "${CLS_OV_PCT}" \
                $RESUME_FLAG \
                CONT.TASK ${task_id} \
                "${extra_args[@]}" \
                SOLVER.BASE_LR 0.0 \
                SOLVER.MAX_ITER ${iter} \
                CONT.COLLECT_QUERY_MODE True \
                SEED $EXP_SEED \
                OUTPUT_DIR "$OUTPUT_DIR"
            
            if [ $? -ne 0 ]; then
                echo " Query collection failed for task ${task_id}!"
                exit 1
            fi
            
            echo "   Query collection completed!"
            echo "  Phase 2: Running continual training..."
           
            
            # Continual training
            NCCL_P2P_LEVEL=LOC python train_continual.py \
                --dist-url auto \
                --num-gpus $ngpus \
                --config-file configs/ade20k/panoptic-segmentation/${SCENARIO}.yaml \
                --img-overlap "${IMG_OV_PCT}" \
                --cls-overlap "${CLS_OV_PCT}" \
                $RESUME_FLAG \
                CONT.TASK ${task_id} \
                "${extra_args[@]}" \
                SOLVER.BASE_LR 0.00005 \
                SOLVER.MAX_ITER 5000 \
                CONT.COLLECT_QUERY_MODE False \
                SEED $EXP_SEED \
                OUTPUT_DIR "$OUTPUT_DIR"
            
            if [ $? -eq 0 ]; then
                echo "   Task ${task_id} completed successfully"
                echo "$(date): Completed task ${task_id}" >> training_log.txt
            else
                echo "   Task ${task_id} failed"
                exit 1
            fi
            
        else
            echo "   Error: Index ${index} out of range for STEP_QUERY_MODE_ITERATIONS array"
            exit 1
        fi
        
        # Brief pause between tasks
        sleep 2
    done
fi

echo ""
echo "All training completed successfully!"
echo "=========================================="
echo "Summary:"
echo "  Scenario: ${SCENARIO}"
echo "  Image Overlap: ${IMG_OV_PCT}%"
echo "  Class Overlap: ${CLS_OV_PCT}%"
echo "  Tasks Completed: Base + ${NUM_TASKS} continual"
echo "  Results Directory: ${RESULT_DIR}"
echo "=========================================="

# Create completion marker
touch "${RESULT_DIR}/ALL_TRAINING_COMPLETE"
echo "$(date): Completed all training for ${OVERLAP_DIR}" >> training_log.txt
