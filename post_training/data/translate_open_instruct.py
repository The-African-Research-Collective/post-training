import os
import re
import yaml
import argparse
import asyncio
import jsonlines
import pandas as pd

from typing import Any
from jinja2 import Template
from datasets import load_dataset
from functools import partial

from post_training.llms.azure_client import AzureOldDeployments
from post_training.llms.base import Generation_Models
from post_training.llms.utils import get_response_format_for_model


def replace_boxed_with_answer(text):
    """
    Replace LaTeX \\boxed{content} with content in the given text.

    Args:
        text (str): The input text containing LaTeX boxed expressions

    Returns:
        str: Text with all boxed expressions replaced with answer tags
    """
    # The regex pattern looks for \\boxed{ followed by any characters (non-greedy) until }
    pattern = r"\\boxed\{([^}]*)\}"

    # Replace with \1 where \1 is the captured content inside the braces
    result = re.sub(pattern, r"\1", text)

    return result


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
        {
            "role": "user",
            "content": prompt_template_dict["user"].render(
                prompt=instance["problem"],
                answer=instance["generated_solution"],
                language=instance["language"],
            ),
        }
    ]


async def main(args):
    llm = AzureOldDeployments(
        deployment_name=args.azure_deployment_name,
        model_name=args.azure_deployment_date,
    )

    generation_kwargs = {"max_tokens": 2048, "structured_object": args.response_class}

    # create a text file for managing processed prompts
    file_path = f"{args.data_directory}/processed_translated_prompts_open_instruct.txt"

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

    input_dataset = load_dataset("json", data_files=args.input_file, split="train")
    input_dataset = input_dataset.shuffle()

    # preprocess dataset response generated_solution text
    input_dataset = input_dataset.map(
        lambda x: {
            "generated_solution": replace_boxed_with_answer(x["generated_solution"])
        },
    )

    with (
        jsonlines.open(
            f"{args.data_directory}/responses_open_instruct_translated.jsonl", mode="a"
        ) as writer,
        open(file_path, "a") as f,
    ):
        for batch in input_dataset.batch(batch_size=args.batch_size):
            batch = pd.DataFrame(batch).to_dict(orient="records")

            filtered_batch = [
                row for row in batch if row["sample_id"] not in processed_pages
            ]
            if len(filtered_batch) == 0:
                continue

            batch_prompts = list(map(prompt_generator, filtered_batch))

            completions = await llm.completion(batch_prompts, **generation_kwargs)

            for row, completion in zip(filtered_batch, completions):
                try:
                    if completion.generation and completion.generation != {}:
                        writer.write(
                            {
                                "id": row["sample_id"],
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": completion.generation[
                                            "problem_translation"
                                        ],
                                    },
                                    {
                                        "role": "assistant",
                                        "content": completion.generation[
                                            "step_by_step_response"
                                        ],
                                    },
                                ],
                                "model": args.model.value,
                                "language": row["language"],
                            }
                        )

                        f.write(row["sample_id"] + "\n")
                except Exception as e:
                    print(f"Error processing row {row['sample_id']}: {e}")
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate math problem response")
    parser.add_argument(
        "--model", type=Generation_Models, choices=list(Generation_Models)
    )
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        default="configs/prompts/math_translation_openinstruct.yaml",
    )
    parser.add_argument(
        "--input_file", type=str, default="files/all_samples_open_instruct.jsonl"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_directory", type=str, default="files/responses")
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
