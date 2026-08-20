#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "Usage: $0 {dpo|grpo|grpo_vllm|sdpo|sdpo_vllm} CONFIG [NUM_GPUS=1] [PRECISION=bf16] [DEEPSPEED_CONFIG]"
    exit 1
fi

ALGORITHM=$1
shift
if [[ $ALGORITHM != "dpo" && $ALGORITHM != "grpo" && $ALGORITHM != "grpo_vllm" && $ALGORITHM != "sdpo" && $ALGORITHM != "sdpo_vllm" ]]; then
    echo "Algorithm must be dpo, grpo, grpo_vllm, sdpo, or sdpo_vllm."
    exit 1
fi
if [[ $ALGORITHM != "dpo" && ${DISTRIBUTED_BACKEND:-auto} =~ ^(fsdp|deepspeed)$ ]]; then
    echo "GRPO and SDPO currently support single GPU or DDP, not FSDP/DeepSpeed."
    exit 1
fi
if [[ $ALGORITHM != "dpo" && $# -ge 4 && -n ${4:-} ]]; then
    echo "GRPO and SDPO cannot use a DeepSpeed config."
    exit 1
fi

TRAINING_MODULE=$ALGORITHM exec "$(dirname "$0")/run_sft.sh" "$@"
