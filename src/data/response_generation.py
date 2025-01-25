"""
"""
import os
import sys
import argparse
import asyncio
import jsonlines

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.persona_generation.prompt_template import MATH_PROMPT_RESPONSE_GENERATION
from src.llms.base import Generation_Models, ModelProvider
from src.llms.litellm_client import LiteLLM
from src.llms.azure_client import AzureOPENAILLM
from src.llms.tgi_inference_client import TGI_client

def _build_prompt_message(
        prompt: str,
        language: str,
        target_domain: str,
        model_name: Generation_Models
):
    if target_domain == "math":
        system_prompt = MATH_PROMPT_RESPONSE_GENERATION
        user_prompt = f"""Math Problem: {prompt}
Language: {language}"""
    else:
        raise ValueError(f"Invalid target domain: {target_domain}")

    if model_name in [Generation_Models.TGI_GEMINI_9B]:
        return [{"role": "user", "content": system_prompt +"\n\n"+ user_prompt}]
    else:
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    

async def main(args):

    if args.model == Generation_Models.AZURE_GPT4O:
        llm = AzureOPENAILLM(model_name=args.model)
    elif args.model in [Generation_Models.TGI_GEMINI_9B]:
        llm = TGI_client(model_name=args.model, model_provider=args.model_provider)
    else:
        llm = LiteLLM(model_name=args.model, model_provider=args.model_provider)

    generation_kwargs = {"max_tokens": 2048 }

    # create a text file for managing processed prompts
    file_path = f"{args.data_directory}/processed_prompts_{args.domain}.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_pages = f.read().splitlines()
    else:
        processed_pages = []
    
    with jsonlines.open(args.prompt_input_file) as reader, open(file_path, "a") as f, jsonlines.open(f"{args.data_directory}/responses_{args.domain}.jsonl", mode="a") as writer:
        for prompt_object in tqdm(reader):
            if prompt_object["id"] in processed_pages:
                continue

            prompt = prompt_object["prompt"]
            language = prompt_object["language"]

            prompt_message = _build_prompt_message(prompt, language, args.domain, args.model)

            completions = await llm.completion([prompt_message],None, **generation_kwargs)

            for completion in completions:
                if completion.generation and completion.generation != {}:
                    writer.write({"id": prompt_object["id"], "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion.generation}], "model": args.model.value, "language": language})

                    f.write(prompt_object["id"] + "\n")

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math problem response")
    parser.add_argument("--model", type=Generation_Models, choices=list(Generation_Models))
    parser.add_argument("--model_provider", type=ModelProvider, choices=list(ModelProvider), required=False)
    parser.add_argument("--prompt_input_file", type=str, required=True)
    parser.add_argument("--data_directory", type=str, default="files/responses")
    parser.add_argument("--domain", type=str, default="instruction_following", help="Domain of the prompt", choices=["instruction_following", "math"])
    args = parser.parse_args()
    asyncio.run(main(args))
