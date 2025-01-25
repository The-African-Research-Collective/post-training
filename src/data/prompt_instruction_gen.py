"""
"""
import os
import asyncio
import argparse
import jsonlines
import random
import pandas as pd

from tqdm import tqdm
from typing import List, Optional
from pydantic import BaseModel

from src.data.persona_generation.prompt_template import (
    PROMPT_GENERATION,
    MATH_PROBLEM_GENERATION,
)
from src.llms.base import Generation_Models, ModelProvider
from src.llms.litellm_client import LiteLLM
from src.llms.azure_client import AzureOPENAILLM
from src.llms.tgi_inference_client import TGI_client
from src.constant import TARGET_LANGUAGES, TARGET_DOMAINS, SUBDOMAINS


SEED_PROMPTS_FILE = "src/data/persona_generation/seed_prompts.jsonl"
CONSTRAINTS_FILE = "src/data/persona_generation/ifeval_instructions.csv"


class GeneratedPrompt(BaseModel):
    prompt: str
    language: str


class GeneratedPromptList(BaseModel):
    prompts: List[GeneratedPrompt]


class Prompt_Persona(BaseModel):
    persona: str
    url: Optional[str] = None
    id: str
    prompt: str
    language: str
    model: str
    domain: Optional[str] = None
    style: Optional[str] = None


def _load_constraints():
    constraints_df = pd.read_csv(CONSTRAINTS_FILE)
    return constraints_df.to_json(orient="records", lines=True)


def _load_seed_prompts():
    with jsonlines.open(SEED_PROMPTS_FILE) as reader:
        seed_prompts_list = list(reader)
    return seed_prompts_list


def _load_personas(args, shuffle=True):
    with jsonlines.open(args.persona_file) as reader:
        personas_list = list(reader)

    if shuffle:
        random.shuffle(personas_list)

    return personas_list


def _build_prompt_message(
    seed_prompt: str,
    seed_language: str,
    persona: dict,
    subdomain: str,
    target_domain: str,
    language: str,
    model_name: Generation_Models,
):
    if target_domain == "math":
        system_prompt = MATH_PROBLEM_GENERATION.replace(
            "{seed_language}", seed_language
        ).replace("{seed_prompt}", seed_prompt)
        user_prompt = f"""Persona: {persona['persona']['persona']}
    Language: {language}"""
    else:
        system_prompt = PROMPT_GENERATION.replace(
            "{seed_language}", seed_language
        ).replace("{seed_prompt}", seed_prompt)

        user_prompt = f"""Persona: {persona['persona']['persona']}
    Domain: {subdomain}
    Style: {target_domain}
    Language: {language}"""

    if model_name in [Generation_Models.TGI_GEMINI_9B]:
        return [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


async def main(args):
    seed_prompts_list = _load_seed_prompts()
    persona_list = _load_personas(args)

    seed_prompts_list = [
        prompt for prompt in seed_prompts_list if prompt["domain"] == args.domain
    ]

    if args.model == Generation_Models.AZURE_GPT4O:
        llm = AzureOPENAILLM(model_name=args.model)
    elif args.model in [Generation_Models.TGI_GEMINI_9B]:
        llm = TGI_client(model_name=args.model, model_provider=args.model_provider)
    else:
        llm = LiteLLM(model_name=args.model, model_provider=args.model_provider)

    # create a text file for managing processed pages
    file_path = f"{args.data_directory}/processed_personas_{args.domain}.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_pages = f.read().splitlines()
    else:
        processed_pages = []

    with jsonlines.open(
        f"{args.data_directory}/generated_prompts_{args.domain}.jsonl", "a"
    ) as writer, open(file_path, "a") as f:
        for persona in tqdm(persona_list):
            if persona["id"] in processed_pages:
                continue

            # Randomly sample a subdomain
            subdomain = random.choice(SUBDOMAINS)
            target_domain = random.choice(TARGET_DOMAINS)
            language = random.choice(TARGET_LANGUAGES)
            sampled_seed_prompt = random.choice(seed_prompts_list)

            if args.domain == "math":
                prompt = _build_prompt_message(
                    sampled_seed_prompt["prompt"],
                    sampled_seed_prompt["language"],
                    persona,
                    subdomain,
                    args.domain,
                    language,
                    args.model,
                )
            else:
                prompt = _build_prompt_message(
                    sampled_seed_prompt["prompt"],
                    sampled_seed_prompt["language"],
                    persona,
                    subdomain,
                    target_domain,
                    language,
                    args.model,
                )

            results = await llm.completion([prompt], GeneratedPromptList)

            try:
                for i in range(len(results[0].generation["prompts"])):
                    prompt_object = Prompt_Persona(
                        persona=persona["persona"]["persona"],
                        url=persona["url"] if "url" in persona else "",
                        id=persona["id"] + f"_{i}",
                        prompt=results[0].generation["prompts"][i]["prompt"],
                        language=language,
                        model=results[0].model,
                        domain=subdomain,
                        style=target_domain,
                    )
                    writer.write(prompt_object.dict())

                processed_pages.append(persona["id"])
                f.write(f"{persona['id']}\n")
            except Exception as e:
                print(f"Error processing {persona} with model {results[0].model}")
                print(e)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Prompts")
    parser.add_argument(
        "--domain",
        type=str,
        default="instruction_following",
        help="Domain of the prompt",
        choices=["instruction_following", "math"],
    )
    parser.add_argument(
        "--model", type=Generation_Models, choices=list(Generation_Models)
    )
    parser.add_argument(
        "--model_provider",
        type=ModelProvider,
        choices=list(ModelProvider),
        required=False,
    )
    parser.add_argument("--persona_file", type=str, help="Persona file")
    parser.add_argument("--data_directory", type=str, default="files/prompts")
    args = parser.parse_args()
    asyncio.run(main(args))
