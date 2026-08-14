from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class DatasetArguments:
    """
    Arguments for dataset configuration for supervised fine-tuning
    """

    chat_template_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "A built-in chat template name or tokenizer id. Uses the model tokenizer template when omitted."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the huggingface dataset to use, the expectation is that this is an existing dataset mixture"
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use"}
    )
    dataset_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The immutable dataset revision to load from the Hub."},
    )
    dataset_format: str = field(
        default="auto",
        metadata={
            "help": "Dataset format: auto, conversational, prompt_completion, text, or pretokenized."
        },
    )
    messages_column: str = field(
        default="messages", metadata={"help": "Column containing role/content messages."}
    )
    text_column: str = field(
        default="text", metadata={"help": "Column containing plain training text."}
    )
    prompt_column: str = field(
        default="prompt", metadata={"help": "Prompt column for prompt/completion data."}
    )
    completion_column: str = field(
        default="completion",
        metadata={"help": "Completion column for prompt/completion data."},
    )
    train_split: str = field(
        default="train", metadata={"help": "Dataset split used for training."}
    )
    dataset_mixer: Optional[dict] = field(
        default=None,
        metadata={"help": "A dictionary of datasets (local or HF) to sample from."},
    )
    dataset_mixer_list: Optional[list[str]] = field(
        default=None,
        metadata={"help": "A list of datasets (local or HF) to sample from."},
    )
    dataset_mix_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to save the mixed dataset to disk."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The path to the training file"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of workers to use for processing the dataset"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, overrides the number of training samples. Otherwise, the dataset size is used."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    language_column: str = field(
        default="language", metadata={"help": "Column used for language filtering."}
    )
    language_subset: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={
            "help": "One language or a list of languages to retain before tokenization."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError(
                "Need either a dataset name, dataset mixer, or a training file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.rsplit(".", maxsplit=1)[-1].lower()
                if extension not in {"json", "jsonl", "csv", "parquet"}:
                    raise ValueError(
                        "`train_file` must be JSON, JSONL, CSV, or Parquet."
                    )
        if (
            (
                self.dataset_name is not None
                and (
                    self.dataset_mixer is not None
                    or self.dataset_mixer_list is not None
                )
            )
            or (self.dataset_name is not None and self.train_file is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None)
                and self.train_file is not None
            )
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")

        supported_formats = {
            "auto",
            "conversational",
            "prompt_completion",
            "text",
            "pretokenized",
        }
        if self.dataset_format not in supported_formats:
            raise ValueError(
                f"Unsupported dataset format {self.dataset_format!r}. "
                f"Choose one of {sorted(supported_formats)}."
            )

        if isinstance(self.language_subset, list) and not self.language_subset:
            raise ValueError("language_subset cannot be an empty list")


@dataclass
class ModelArguments:
    """
    Arguments for model configuration.
    """

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The version of the model on huggingface to use"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The model configuration to use, it is usually a model name"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to execute custom code from a remote model repo."},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer to use"}
    )
    tokenizer_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer revision. Defaults to the model revision."},
    )
    use_slow_tokenizer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a slow tokenizer"}
    )
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use the LORA training"}
    )
    lora_rank: Optional[int] = field(
        default=64, metadata={"help": "The rank of the LORA model"}
    )
    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    use_qlora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use the qlora training"}
    )
    add_bos_token: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add a beginning of sentence token"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Model dtype: auto, float32, float16, or bfloat16."},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Transformers attention backend, for example eager, sdpa, or flash_attention_2."
        },
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "LoRA target modules. Uses common projection modules when omitted."
        },
    )

    def __post_init__(self):
        supported_dtypes = {"auto", "float32", "float16", "bfloat16"}
        if self.torch_dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported torch_dtype {self.torch_dtype!r}. "
                f"Choose one of {sorted(supported_dtypes)}."
            )
        if self.attn_implementation not in {
            None,
            "eager",
            "sdpa",
            "flash_attention_2",
        }:
            raise ValueError(
                "attn_implementation must be eager, sdpa, flash_attention_2, or null"
            )
        if self.use_qlora and not self.use_lora:
            raise ValueError("use_qlora requires use_lora")


@dataclass
class ExperimentArguments:
    """
    Arguments for experiment configuration.
    """

    exp_name: str = field(metadata={"help": "The name of the experiment"})
    run_name: str = field(default=None, metadata={"help": "The name of the run"})
    project_name: str = field(
        default=None, metadata={"help": "The name of the project"}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether to push the model to the hub"}
    )
    with_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of gradient accumulation steps to use."},
    )
    output_dir: str = field(
        default="output/",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."},
    )
    report_to: Union[str, List[str]] = field(
        default="none",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."
        },
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory. Means that resumption will always start from scratch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3,
        metadata={
            "help": "How many checkpoints to keep in the output directory. -1 for all."
        },
    )
    fused_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use fused AdamW or not.",
        },
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={
            "help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Entity to use for logging to wandb."},
    )
    wandb_project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project name to use for logging to wandb."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    hf_repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "The huggingface repository id to push the model to"},
    )
    hf_entity: Optional[str] = field(
        default=None, metadata={"help": "The huggingface entity to push the model to"}
    )
    hf_repo_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The huggingface repository revision to push the model to"},
    )
    hf_private_repo: bool = field(
        default=True,
        metadata={"help": "Create a private Hugging Face repository when needed."},
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "The seed to use for the run"}
    )
    timeout: int = field(
        default=600, metadata={"help": "The timeout for the run"}
    )
    use_flash_attention: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use flash attention"}
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."
        },
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    load_balancing_loss: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to include a load balancing loss used in mixture of experts training.",
        },
    )
    load_balancing_weight: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "The weight of the load balancing loss used in mixture of experts training.",
        },
    )
    logging_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Log every n steps."},
    )
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Save checkpoint every n steps."},
    )
    mask_instructions: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to mask the instructions in the training data"},
    )

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive when provided")
        if self.keep_last_n_checkpoints < -1:
            raise ValueError("keep_last_n_checkpoints must be -1 or greater")
        report_targets = (
            [self.report_to] if isinstance(self.report_to, str) else self.report_to
        )
        if self.with_tracking and (not report_targets or report_targets == ["none"]):
            raise ValueError("with_tracking requires at least one report_to integration")
