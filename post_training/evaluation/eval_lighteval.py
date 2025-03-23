import os
import yaml
import torch
import argparse
from dotenv import load_dotenv
from datetime import timedelta

from lighteval.models.model_input import GenerationParameters
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.transformers.adapter_model import AdapterModelConfig
from lighteval.models.transformers.delta_model import DeltaModelConfig
from lighteval.models.transformers.transformers_model import (
    BitsAndBytesConfig,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))]
    )
else:
    accelerator = None

EVAL_PATH = "files/evaluation"
TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR: str = os.getenv("HF_HOME")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

os.makedirs(EVAL_PATH, exist_ok=True)
load_dotenv()


def main(args):
    evaluation_tracker = EvaluationTracker(
        output_dir=EVAL_PATH,
        save_details=True,
        push_to_hub=True,
        hub_results_org="taresco",
        push_to_tensorboard=False,
        public=True,
    )

    env_config = EnvConfig(token=TOKEN, cache_dir=CACHE_DIR)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=env_config,
        override_batch_size=args.override_batch_size,
        max_samples=args.max_samples,
        custom_tasks_directory=CURRENT_DIR + "/afrimgsm_evals.py",
    )

    # Load model configuration
    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)["model"]

    if args.inference_type == "vllm":
        model_args = config["base_params"]["model_args"]
        generation_parameters = GenerationParameters.from_dict(config)

        model_args_dict: dict = {
            k.split("=")[0]: k.split("=")[1] if "=" in k else True
            for k in model_args.split(",")
        }
        model_config = VLLMModelConfig(
            **model_args_dict, generation_parameters=generation_parameters
        )
    elif args.inference_type == "accelerate":
        # Creating optional quantization configuration
        if config["base_params"]["dtype"] == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        elif config["base_params"]["dtype"] == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        # We extract the model args
        args_dict = {
            k.split("=")[0]: k.split("=")[1]
            for k in config["base_params"]["model_args"].split(",")
        }

        args_dict["generation_parameters"] = GenerationParameters.from_dict(config)

        # We store the relevant other args
        args_dict["base_model"] = config["merged_weights"]["base_model"]
        args_dict["compile"] = bool(config["base_params"]["compile"])
        args_dict["dtype"] = config["base_params"]["dtype"]
        args_dict["accelerator"] = accelerator
        args_dict["quantization_config"] = quantization_config
        args_dict["batch_size"] = args.override_batch_size
        args_dict["multichoice_continuations_start_space"] = config["base_params"][
            "multichoice_continuations_start_space"
        ]
        args_dict["use_chat_template"] = args.use_chat_template

        # Keeping only non null params
        args_dict = {k: v for k, v in args_dict.items() if v is not None}

        if config["merged_weights"].get("delta_weights", False):
            if config["merged_weights"]["base_model"] is None:
                raise ValueError(
                    "You need to specify a base model when using delta weights"
                )
            model_config = DeltaModelConfig(**args_dict)
        elif config["merged_weights"].get("adapter_weights", False):
            if config["merged_weights"]["base_model"] is None:
                raise ValueError(
                    "You need to specify a base model when using adapter weights"
                )
            model_config = AdapterModelConfig(**args_dict)
        elif config["merged_weights"]["base_model"] not in ["", None]:
            raise ValueError(
                "You can't specify a base model if you are not using delta/adapter weights"
            )
        else:
            model_config = TransformersModelConfig(**args_dict)

    task = args.task

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a task using lighteval."
    )
    parser.add_argument(
        "--task", type=str, help="The task to evaluate the model on.", required=True
    )
    parser.add_argument(
        "--model_config",
        type=str,
        help="The model configuration to use for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--inference_type",
        type=str,
        help="The type of inference to use for evaluation.",
        choices=["accelerate", "vllm"],
        required=True,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="The maximum number of samples to evaluate on.",
        default=None,
    )
    parser.add_argument(
        "--override_batch_size",
        type=int,
        help="The batch size to use for evaluation.",
        default=1,
    )
    parser.add_argument(
        "--use_chat_template",
        type=bool,
        help="Whether to use the chat template for the model.",
        default=True,
    )
    args = parser.parse_args()
    main(args)
