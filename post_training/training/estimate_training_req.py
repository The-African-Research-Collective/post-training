import argparse

from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from transformers import AutoModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_gpus_per_node", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=True)
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_name)
    estimate_zero3_model_states_mem_needs_all_live(
        model, num_gpus_per_node=args.num_gpus_per_node, num_nodes=args.num_nodes
    )
