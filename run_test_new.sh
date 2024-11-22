#!/bin/bash

# 配置参数
PRETRAINED_MODEL_PATH="../runwayml-stable-diffusion-v1-5"
OUTPUT_DIR="sd-model-fine-tuned-new1"
TRAIN_BATCH_SIZE=16
CHECKPOINTS=("checkpoint-1000" "checkpoint-5000" "checkpoint-10000" "checkpoint-20000" "checkpoint-30000")  # 假设你有多个检查点
SCRIPT_NAME="test.py"
NUM_PROCESSES=2
MAX_BATCH=1

# 遍历检查点
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    # 构建命令
    COMMAND="accelerate launch --num_processes $NUM_PROCESSES $SCRIPT_NAME \
        --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
        --resume_from_checkpoint $CHECKPOINT \
        --output_dir $OUTPUT_DIR \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --max_infer_batch $MAX_BATCH \
        --use_ema"

    # 显示当前命令
    echo "Running command: $COMMAND"

    # 执行命令
    $COMMAND
done
