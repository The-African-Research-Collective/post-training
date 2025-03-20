"""
python -m post_training.data.response_generation \
    --batch_size 1 \
    --deployment_name gpt-4o-meenee \
    --model azure_ai/newgpt4o \
    --prompt_template_file configs/prompts/v2_prompts.yaml \
    --prompt_input_file data/translated_grade_school_prompts.jsonl \
    --data_directory data/gpt4omini_generationst_translated_prompt \
    --domain math \
    --response_class "answers.Answer"
"""

import os
import argparse
import asyncio
from functools import partial
from typing import Any
from datetime import datetime

import pandas as pd
import jsonlines
import yaml
from datasets import load_dataset
from jinja2 import Template

from post_training.llms.azure_client import AzureOPENAILLM, AzureOldDeployments
from post_training.llms.base import Generation_Models, ModelProvider
from post_training.llms.litellm_client import LiteLLM
from post_training.llms.tgi_inference_client import TGI_client
from post_training.llms.utils import get_response_format_for_model


def _build_prompt_message(
    instance: dict[str, Any],
    model_name: Generation_Models,
    prompt_template_dict: dict[str, Template],
) -> list[dict[str, str]]:
    if model_name == Generation_Models.TGI_GEMINI_9B:
        return [
            {
                "role": "user",
                "content": prompt_template_dict["system"].render(instance)
                + "\n\n"
                + prompt_template_dict["user"].render(instance)
                if "system" in prompt_template_dict
                else prompt_template_dict["user"].render(instance),
            }
        ]

    conversation = []
    if "system" in prompt_template_dict:
        conversation.append(
            {
                "role": "system",
                "content": prompt_template_dict["system"].render(instance),
            }
        )

    return conversation + [
        {"role": "user", "content": prompt_template_dict["user"].render(instance)}
    ]


async def main(args):
    if args.model == Generation_Models.AZURE_GPT4O:
        # Check if model deployment data is provided and earlier than "2024-08-01"  or not
        if args.azure_deployment_date and datetime.strptime(
            args.azure_deployment_date, "%Y-%m-%d"
        ) < datetime.strptime("2024-08-01", "%Y-%m-%d"):
            llm = AzureOldDeployments(
                deployment_name=args.azure_deployment_name,
                model_name=args.azure_deployment_date,
            )
        else:
            llm = AzureOPENAILLM(
                deployment_name=args.azure_deployment_name, model_name=args.model
            )
    elif args.model in [Generation_Models.TGI_GEMINI_9B]:
        llm = TGI_client(model_name=args.model, model_provider=args.model_provider)
    else:
        llm = LiteLLM(model_name=args.model, model_provider=args.model_provider)

    generation_kwargs = {"max_tokens": 2048, "structured_object": args.response_class}

    # create a text file for managing processed prompts
    file_path = f"{args.data_directory}/processed_prompts_{args.domain}.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_pages = set(f.read().splitlines())
    else:
        processed_pages = set()

    with open(args.prompt_template_file, "r") as stream:
        prompt_template_dict = yaml.safe_load(stream)

        if "system" in prompt_template_dict:
            prompt_template_dict["system"] = Template(prompt_template_dict["system"])

        assert "user" in prompt_template_dict, (
            "prompt template must contain a user template"
        )
        prompt_template_dict["user"] = Template(prompt_template_dict["user"])

    prompt_generator = partial(
        _build_prompt_message,
        model_name=args.model,
        prompt_template_dict=prompt_template_dict,
    )

    input_dataset = load_dataset(
        "json", data_files=args.prompt_input_file, split="train"
    )

    with (
        jsonlines.open(
            f"{args.data_directory}/responses_{args.domain}.jsonl", mode="a"
        ) as writer,
        open(file_path, "a") as f,
    ):
        for batch in input_dataset.batch(batch_size=args.batch_size):
            batch = pd.DataFrame(batch).to_dict(orient="records")

            filtered_batch = [row for row in batch if row["id"] not in processed_pages]
            if len(filtered_batch) == 0:
                continue

            batch_prompts = list(map(prompt_generator, filtered_batch))

            completions = await llm.completion(batch_prompts, **generation_kwargs)

            for row, completion in zip(filtered_batch, completions):
                if completion.generation and completion.generation != {}:
                    writer.write(
                        {
                            "id": row["id"],
                            "messages": [
                                {"role": "user", "content": row["content"]},
                                {"role": "assistant", "content": completion.generation},
                            ],
                            "model": args.model.value,
                            "language": row["language"],
                        }
                    )

                    f.write(row["id"] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math problem response")
    parser.add_argument(
        "--deployment_name", type=str, help="Azure OpenAI deployment name"
    )
    parser.add_argument(
        "--model", type=Generation_Models, choices=list(Generation_Models)
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--model_provider",
        type=ModelProvider,
        choices=list(ModelProvider),
        required=False,
    )
    parser.add_argument(
        "--prompt_template_file", type=str, default="configs/prompts/v1_prompts.yaml"
    )
    parser.add_argument("--prompt_input_file", type=str, required=True)
    parser.add_argument("--data_directory", type=str, default="files/responses")
    parser.add_argument(
        "--domain",
        type=str,
        default="instruction_following",
        help="Domain of the prompt",
        choices=["instruction_following", "math"],
    )
    parser.add_argument(
        "--response_class",
        type=str,
        help="Fully-qualified class name: module.class_name. "
        "The module will be dynamically imported so ensure it is placed in the root directory where this script is invoked.",
    )
    parser.add_argument("--azure_deployment_name", type=str, required=False)
    parser.add_argument("--azure_deployment_date", type=str, required=False)
    args = parser.parse_args()

    if args.response_class is not None:
        args.response_class = get_response_format_for_model(args.response_class)

    asyncio.run(main(args))
