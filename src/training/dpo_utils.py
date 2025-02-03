from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from datasets import load_dataset

from src.constants import TOKENIZED_PREFERENCE_DATASET_KEYS
from src.sft_utils import encode_sft_example


# Dataset Configuration and Caching
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_split: str
    dataset_revision: str
    dataset_range: Optional[int] = None
    transform_fn: List[str] = field(default_factory=list)
    transform_fn_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # for tracking purposes
    dataset_commit_hash: Optional[str] = None

    def __post_init__(self):
        self.dataset = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            revision=self.dataset_revision,
        )
        if self.dataset_range is None:
            dataset_range = len(self.dataset)
            self.update_range(dataset_range)

    def update_range(self, dataset_range: int):
        self.dataset_range = dataset_range
        if self.dataset_range > len(self.dataset):
            raise ValueError("Dataset range exceeds dataset length")
        self.dataset = self.dataset.select(range(self.dataset_range))
    
def tokenize_and_truncate():
    pass

def filter_dataset():
    pass

def get_cached_preference_dataset(dataset_mixer_list: List[str],
                tokenizer: AutoTokenizer,
                max_seq_length: int) -> None:
    
    dcs = []
    assert len(dataset_mixer_list) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer_list}"

    for i in range(0, len(dataset_mixer_list), 2):

        dataset_name = dataset_mixer_list[i]
        frac_or_num_samples = dataset_mixer_list[i + 1]
        if "." in frac_or_num_samples:
            # Fraction of the dataset
            frac_or_num_samples = float(frac_or_num_samples)
        else:
            # Number of samples
            frac_or_num_samples = int(frac_or_num_samples)
        
        dataset_config = DatasetConfig(
            dataset_name=dataset_name,
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["tokenize_and_truncate", "filter_dataset"],
            transform_fn_args={
                "tokenize_and_truncate": {
                    "max_seq_length": max_seq_length,
                    "target_columns": TOKENIZED_PREFERENCE_DATASET_KEYS,
                }
            },
        )
        if frac_or_num_samples > 1.0:
            new_range = int(frac_or_num_samples)
        else:
            new_range = int(frac_or_num_samples * len(dataset_config.dataset))
        
        dataset_config.update_range(new_range)
        dcs.append(dataset_config)
    


    



