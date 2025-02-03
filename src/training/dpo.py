import os
import time
import logging
import datasets
import transformers
import torch
import deepspeed
import functools
import random
import math
import json

from datetime import timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Union
from accelerate.utils import InitProcessGroupKwargs, set_seed, DataLoaderConfiguration
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    DataCollatorForSeq2Seq,
    OPTForCausalLM,
    get_scheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    ArgumentParserPlus,
    mix_datasets,
    CHAT_TEMPLATES,
    get_last_checkpoint_path,
    clean_last_n_checkpoints,
    upload_metadata_to_hf,
    push_folder_to_hub,
)
from model_utils import save_with_accelerate
from src.training.sft_args import ExperimentArguments, ModelArguments, DatasetArguments

logger = get_logger(__name__)

def main(args):
    exp_args, model_args, data_args = args[0], args[1], args[2]

    exp_args.output_dir = os.path.join(exp_args.output_dir, exp_args.exp_name)

    exp_args.run_name = f"{exp_args.exp_name}__{exp_args.seed}__{int(time.time())}"
    if exp_args.push_to_hub:
        exp_args.run_name = f"{exp_args.run_name}__hub"

        if exp_args.hf_entity is None:
            exp_args.hf_entity = "taresco"
        if exp_args.hf_repo_id is None:
            exp_args.hf_repo_id = f"{exp_args.hf_entity}/{exp_args.exp_name}"
        if exp_args.hf_repo_revision is None:
            exp_args.hf_repo_revision = exp_args.run_name

        exp_args.hf_repo_url = f"https://huggingface.co/{exp_args.hf_repo_id}/tree/{exp_args.hf_repo_revision}"
    
    accelerator_log_kwargs = {}

    if exp_args.with_tracking:
        accelerator_log_kwargs["log_with"] = exp_args.report_to
        accelerator_log_kwargs["project_dir"] = exp_args.output_dir
    
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=exp_args.timeout))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    accelerator = Accelerator(
        gradient_accumulation_steps=exp_args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    if exp_args.seed:
        set_seed(exp_args.seed)
    
    # Create output directory in the main process
    if accelerator.is_main_process:
        if exp_args.output_dir is not None:
            os.makedirs(exp_args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    # load tokenizer
    tokenizer_revision = (
        model_args.tokenizer_revision
        if model_args.tokenizer_revision
        else model_args.model_revision
    )
    if tokenizer_revision != model_args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{model_args.model_revision}`."""
        logger.warning(warning)

    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        use_fast=not exp_args.use_slow_tokenizer,
        trust_remote_code=model_args.trust_remote_code,
    )

    if data_args.dataset_mixer:
        data_args.dataset_mixer_list = [item for pair in data_args.dataset_mixer.items() for item in pair]



    


if __name__ == "__main__":
    parser = ArgumentParserPlus((ExperimentArguments, ModelArguments, DatasetArguments))
    args = parser.parse()
    main(args)
