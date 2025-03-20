"""
python3 src/data/create_personas_wura.py --batch_size 10 --language eng \
    --model "azure_ai/newgpt4o"
"""

import os
import time
import asyncio
import argparse
from datetime import datetime

import jsonlines
import tenacity
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

from post_training.data.persona_generation.prompt_template import PERSONA_GENERATION
from post_training.llms.azure_client import AzureOPENAILLM, AzureOldDeployments
from post_training.llms.base import Generation_Models, ModelProvider
from post_training.llms.litellm_client import LiteLLM
from post_training.llms.tgi_inference_client import TGI_client

load_dotenv()

DATASET_NAME = "castorini/wura"


class Persona(BaseModel):
    countries: list[str]
    languages: list[str]
    persona: str


class PersonaList(BaseModel):
    persona_list: list[Persona]


def prompt_processor(example):
    # get the first 200 words of the text
    text = example["content"].split()[:200]
    text = " ".join(text)
    prompt = [
        {
            "role": "system",
            "content": PERSONA_GENERATION,
            "cache_control": {"type": "ephemeral"},
        },
        {"role": "user", "content": text},
    ]

    example["templated_prompt"] = prompt
    return example


def batch_dataset_generator(dataset, batch_size):
    # Create prompt
    dataset = dataset.map(prompt_processor)
    # Shuffle the dataset
    dataset = dataset.shuffle()
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


async def main(args):
    dataset = load_dataset(DATASET_NAME, f"{args.language}")

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
        llm = TGI_client(model_name=args.model)
    else:
        llm = LiteLLM(model_name=args.model, model_provider=args.model_provider)

    # create a text file for managing processed pages
    file_path = f"{args.data_directory}/{args.language}_processed_urls.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_pages = f.read().splitlines()
    else:
        processed_pages = []

    # filter dataset to pages that have more than 1000 words
    dataset = dataset.filter(
        lambda x: len(x["content"].split()) > 100 if x["content"] else False
    )
    print(f"Processing {len(dataset['train'])} pages")

    # Remove pages that have already been processed
    dataset = dataset.filter(lambda x: str(x["id"]) not in set(processed_pages))
    print(f"Processing {len(dataset['train'])} pages")

    lang_output_path = f"{args.data_directory}"
    os.makedirs(lang_output_path, exist_ok=True)

    with (
        jsonlines.open(
            f"{args.data_directory}/{args.language}_personas.jsonl", "a"
        ) as writer,
        open(file_path, "a") as f,
    ):
        # iterate over the dataset and use multiprocessing to generate personas

        for i, batch in tqdm(
            enumerate(batch_dataset_generator(dataset["train"], args.batch_size)),
            total=len(dataset) // args.batch_size,
        ):
            print(f"Processing batch {i}")

            try:
                results = await llm.completion(batch["templated_prompt"], PersonaList)

                for i, id, url, res in zip(
                    range(len(batch)), batch["id"], batch["url"], results
                ):
                    model_used = res.model
                    try:
                        for persona in res.generation["persona_list"]:
                            writer.write(
                                {
                                    "id": id,
                                    "url": url,
                                    "persona": persona,
                                    "model": model_used,
                                }
                            )
                        f.write(f"{id}\n")
                    except KeyError:
                        print(f"Error processing {id} with model {model_used}")

            except tenacity.RetryError:
                print(f"Error processing batch {i}")
                continue

            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default="files/wura_personas")
    parser.add_argument("--language", type=str)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--model", type=Generation_Models, choices=list(Generation_Models)
    )
    parser.add_argument(
        "--model_provider",
        type=ModelProvider,
        choices=list(ModelProvider),
        required=False,
    )
    parser.add_argument("--azure_deployment_name", type=str, required=False)
    parser.add_argument("--azure_deployment_date", type=str, required=False)
    args = parser.parse_args()

    asyncio.run(main(args))
