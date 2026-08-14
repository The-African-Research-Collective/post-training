#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 CONFIG [NUM_GPUS=1] [PRECISION=bf16] [DEEPSPEED_CONFIG]"
    exit 1
fi

CONFIG=$1
NUM_GPUS=${2:-1}
TRAINING_PRECISION=${3:-bf16}
DEEPSPEED_CONFIG=${4:-}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}

# Accelerate calls full precision "no". Keep fp32 as a readable alias.
if [[ $TRAINING_PRECISION == "fp32" ]]; then
    ACCELERATE_PRECISION=no
elif [[ $TRAINING_PRECISION == "no" || $TRAINING_PRECISION == "bf16" || $TRAINING_PRECISION == "fp16" ]]; then
    ACCELERATE_PRECISION=$TRAINING_PRECISION
else
    echo "Invalid precision: $TRAINING_PRECISION. Choose from 'fp32', 'no', 'fp16', or 'bf16'."
    exit 1
fi

if [[ ! $NUM_GPUS =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer."
    exit 1
fi

if [[ ! -f $CONFIG ]]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

COMMAND=(
    accelerate launch
    --mixed_precision "$ACCELERATE_PRECISION"
    --num_machines 1
    --num_processes "$NUM_GPUS"
    --main_process_port "$MAIN_PROCESS_PORT"
)

if [[ -n $DEEPSPEED_CONFIG ]]; then
    if [[ ! -f $DEEPSPEED_CONFIG ]]; then
        echo "DeepSpeed config not found: $DEEPSPEED_CONFIG"
        exit 1
    fi
    COMMAND+=(--use_deepspeed --deepspeed_config_file "$DEEPSPEED_CONFIG")
elif (( NUM_GPUS > 1 )); then
    COMMAND+=(--multi_gpu)
fi

echo "Training with $NUM_GPUS GPU process(es); CUDA visibility is inherited."
exec "${COMMAND[@]}" -m post_training.training.sft "$CONFIG"
