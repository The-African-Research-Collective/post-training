
CONFIG=$1
NUM_GPUS=$2
TRAINING_PRECISION=$3

echo "Training model using $NUM_GPUS GPUs"

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Check that precision exists among the available list of options ("fp32", "bf16", "fp16")
if [[ $TRAINING_PRECISION != "fp32" && $TRAINING_PRECISION != "bf16" && $TRAINING_PRECISION != "fp16" ]]; then
    echo "Invalid training precision. Please choose from 'fp32', 'bf16', or 'fp16'."
    exit 1
fi


accelerate launch \
    --mixed_precision $TRAINING_PRECISION \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 29501 \
    --deepspeed_config_file configs/deep_speed/stage3_offloading_accelerate.conf \
    post_training/training/sft.py $CONFIG