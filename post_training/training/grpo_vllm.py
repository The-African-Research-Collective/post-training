"""Run the low-level GRPO objective with vLLM rollout sampling."""

import logging

from post_training.training.alignment_utils import parse_alignment_args
from post_training.training.grpo import train


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train(parse_alignment_args(), rollout_backend="vllm")


if __name__ == "__main__":
    main()
