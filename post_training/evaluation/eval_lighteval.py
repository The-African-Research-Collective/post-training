import os
import argparse
from datetime import timedelta

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
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

os.makedirs(EVAL_PATH, exist_ok=True)


def main(args):
    evaluation_tracker = EvaluationTracker(
        output_dir=EVAL_PATH,
        save_details=True,
        push_to_hub=True,
        hub_results_org="taresco",
        push_to_tensorboard=False,
        public=True,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
        override_batch_size=args.override_batch_size,
        max_samples=args.max_samples,
    )

    model_config = VLLMModelConfig(
        pretrained="HuggingFaceH4/zephyr-7b-beta",
        dtype="float16",
        use_chat_template=True,
    )

    task = args.task

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        custom_task_directory=None,  # if using a custom task
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a task using lighteval."
    )
    parser.add_argument("--task", type=str, help="The task to evaluate the model on.")
    parser.add_argument(
        "--model_config",
        type=str,
        help="The model configuration to use for evaluation.",
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
    args = parser.parse_args()
    main(args)
