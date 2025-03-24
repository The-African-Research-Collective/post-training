#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# Usage description
usage() {
    echo "Usage: $0 [--python PYTHON_EXEC] --tasks COMMA_SEPARATED_TASKS --model_config CONFIG_PATH"
    echo "Example: $0 --tasks 'community|task1,community|task2' --model_config configs/evaluation/tar"
    exit 1
}

# Initialize variables
PYTHON_EXEC="python3"
TASKS=()
MODEL_CONFIG=""

# Parse command line arguments
PARSED_ARGUMENTS=$(getopt -o p:t:c: --long python:,tasks:,model_config: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
    usage
fi

eval set -- "$PARSED_ARGUMENTS"

while :
do
    case "$1" in
        -p | --python)      PYTHON_EXEC="$2"; shift 2 ;;
        -t | --tasks)       IFS=',' read -ra TASKS <<< "$2"; shift 2 ;;
        -c | --model_config) MODEL_CONFIG="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unexpected option: $1"; usage ;;
    esac
done

# Validate required parameters
if [ ${#TASKS[@]} -eq 0 ] || [ -z "$MODEL_CONFIG" ]; then
    echo "Error: --tasks and --model_config are required parameters"
    usage
fi

# Build command arguments
CMD_ARGS=()
for task in "${TASKS[@]}"; do
    CMD_ARGS+=("--task" "$task")
done

# Execute the command
$PYTHON_EXEC -m post_training.evaluation.eval_lighteval \
    "${CMD_ARGS[@]}" \
    --model_config "$MODEL_CONFIG" \
    --inference_type "accelerate" 