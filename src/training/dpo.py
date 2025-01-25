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
