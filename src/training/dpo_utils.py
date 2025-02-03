from typing import List

from transformers import AutoTokenizer

def cached_dpo_dataset(dataset_mixer_list: List[str], tokenizer: AutoTokenizer)