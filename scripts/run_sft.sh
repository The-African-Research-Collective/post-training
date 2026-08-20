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
DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND:-auto}
FSDP_SHARDING_STRATEGY=${FSDP_SHARDING_STRATEGY:-FULL_SHARD}
TRAINING_MODULE=${TRAINING_MODULE:-sft}

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

if [[ $DISTRIBUTED_BACKEND == "auto" ]]; then
    if [[ -n $DEEPSPEED_CONFIG ]]; then
        DISTRIBUTED_BACKEND=deepspeed
    elif (( NUM_GPUS > 1 )); then
        DISTRIBUTED_BACKEND=ddp
    else
        DISTRIBUTED_BACKEND=single
    fi
fi

if [[ $DISTRIBUTED_BACKEND == "deepspeed" ]]; then
    if [[ -z $DEEPSPEED_CONFIG ]]; then
        echo "DISTRIBUTED_BACKEND=deepspeed requires a DeepSpeed config as the fourth argument."
        exit 1
    fi
    if [[ ! -f $DEEPSPEED_CONFIG ]]; then
        echo "DeepSpeed config not found: $DEEPSPEED_CONFIG"
        exit 1
    fi
    COMMAND+=(--use_deepspeed --deepspeed_config_file "$DEEPSPEED_CONFIG")
elif [[ -n $DEEPSPEED_CONFIG ]]; then
    echo "A DeepSpeed config can only be used with DISTRIBUTED_BACKEND=auto or deepspeed."
    exit 1
elif [[ $DISTRIBUTED_BACKEND == "fsdp" ]]; then
    if (( NUM_GPUS < 2 )); then
        echo "DISTRIBUTED_BACKEND=fsdp requires at least two GPUs."
        exit 1
    fi
    case $FSDP_SHARDING_STRATEGY in
        FULL_SHARD|SHARD_GRAD_OP|HYBRID_SHARD|HYBRID_SHARD_ZERO2) ;;
        *)
            echo "Invalid FSDP_SHARDING_STRATEGY: $FSDP_SHARDING_STRATEGY"
            exit 1
            ;;
    esac
    COMMAND+=(
        --use_fsdp
        --fsdp_sharding_strategy "$FSDP_SHARDING_STRATEGY"
        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
        --fsdp_state_dict_type FULL_STATE_DICT
        --fsdp_use_orig_params true
        --fsdp_cpu_ram_efficient_loading true
        --fsdp_sync_module_states true
    )
elif [[ $DISTRIBUTED_BACKEND == "ddp" ]]; then
    if (( NUM_GPUS < 2 )); then
        echo "DISTRIBUTED_BACKEND=ddp requires at least two GPUs."
        exit 1
    fi
    COMMAND+=(--multi_gpu)
elif [[ $DISTRIBUTED_BACKEND != "single" ]]; then
    echo "Invalid DISTRIBUTED_BACKEND: $DISTRIBUTED_BACKEND. Choose auto, single, ddp, deepspeed, or fsdp."
    exit 1
fi

echo "Training $TRAINING_MODULE with $NUM_GPUS GPU process(es) using $DISTRIBUTED_BACKEND; CUDA visibility is inherited."
exec "${COMMAND[@]}" -m "post_training.training.$TRAINING_MODULE" "$CONFIG"
