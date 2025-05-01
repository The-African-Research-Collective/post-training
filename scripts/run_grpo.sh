CONFIG=$1
NUM_GPUS=$2
TRAINING_PRECISION=$3
BASE_MODEL=$4

# Check that precision exists among the available list of options ("fp32", "bf16", "fp16")
if [[ $TRAINING_PRECISION != "fp32" && $TRAINING_PRECISION != "bf16" && $TRAINING_PRECISION != "fp16" ]]; then
    echo "Invalid training precision. Please choose from 'fp32', 'bf16', or 'fp16'."
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model $BASE_MODEL

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --use_deepspeed \
    ---deepspeed_config_file configs/deep_speed/stage3_offloading_accelerate.conf \
    --num_processes 7 \
    post_training/training/grpo.py --config $CONFIG