"""Configuration shared by the low-level DPO, GRPO, and SDPO trainers."""

from dataclasses import dataclass, field


@dataclass
class AlignmentArguments:
    """One readable configuration surface for offline and online alignment."""

    # Model
    model_name_or_path: str
    model_revision: str = "main"
    reference_model_name_or_path: str | None = None
    reference_model_revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # Dataset
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    dataset_revision: str | None = None
    dataset_train_split: str = "train"
    train_file: str | None = None
    max_train_samples: int | None = None
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    solution_column: str = "solution"
    feedback_column: str = "privileged_context"
    system_prompt: str | None = None

    # Training
    exp_name: str = "alignment"
    run_name: str | None = None
    output_dir: str = "runs"
    seed: int = 42
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 1
    max_train_steps: int | None = None
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    checkpointing_steps: str | None = "epoch"
    keep_last_n_checkpoints: int = 3
    resume_from_checkpoint: str | None = None
    overwrite_output_dir: bool = False
    timeout: int = 1800

    # Sequence lengths
    max_length: int = 1024
    max_prompt_length: int = 512
    max_completion_length: int = 256

    # DPO and reference regularization
    beta: float = 0.1
    label_smoothing: float = 0.0

    # GRPO rollout and clipped policy objective
    num_generations: int = 4
    temperature: float = 0.9
    top_p: float = 1.0
    epsilon: float = 0.2
    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy"])
    reward_weights: list[float] | None = None
    cosine_min_value_wrong: float = 0.0
    cosine_max_value_wrong: float = -0.5
    cosine_min_value_correct: float = 0.5
    cosine_max_value_correct: float = 1.0
    cosine_max_len: int = 1000
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0

    # Optional vLLM rollout server used by grpo_vllm and sdpo_vllm
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_group_port: int = 51216
    vllm_server_timeout: float = 300.0
    vllm_repetition_penalty: float = 1.0
    vllm_top_k: int = -1
    vllm_min_p: float = 0.0
    vllm_gpu_memory_utilization: float = 0.85
    vllm_enforce_eager: bool = False

    # SDPO sampled-token self-distillation
    distillation_weight: float = 1.0
    success_reward_threshold: float = 1.0
    use_successful_as_teacher: bool = True
    include_environment_feedback: bool = False
    dont_reprompt_on_self_success: bool = True
    reprompt_template: str = (
        "{prompt}\n\nA previous attempt received feedback. Use it to improve the "
        "answer.\n{solution}{feedback}"
    )
    solution_template: str = (
        "Successful previous attempt:\n{successful_previous_attempt}\n"
    )
    feedback_template: str = "Environment feedback:\n{feedback_raw}\n"

    # Tracking and publication
    with_tracking: bool = False
    report_to: list[str] = field(default_factory=list)
    project_name: str | None = None
    trackio_project_name: str | None = None
    trackio_space_id: str | None = None
    wandb_entity: str | None = None
    push_to_hub: bool = False
    hf_repo_id: str | None = None
    hf_repo_revision: str = "main"
    hf_private_repo: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.report_to, str):
            self.report_to = [item for item in self.report_to.split(",") if item]
        if isinstance(self.reward_funcs, str):
            self.reward_funcs = [item for item in self.reward_funcs.split(",") if item]
        if not self.dataset_name and not self.train_file:
            raise ValueError("Set dataset_name or train_file")
        if self.dataset_name and self.train_file:
            raise ValueError("Set only one of dataset_name or train_file")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if (
            not 0 < self.vllm_server_port <= 65535
            or not 0 < self.vllm_group_port <= 65535
        ):
            raise ValueError("vLLM ports must be between 1 and 65535")
        if self.vllm_server_port == self.vllm_group_port:
            raise ValueError("vLLM server_port and group_port must differ")
        if self.vllm_server_timeout <= 0:
            raise ValueError("vllm_server_timeout must be positive")
        if self.vllm_repetition_penalty <= 0:
            raise ValueError("vllm_repetition_penalty must be positive")
        if self.vllm_top_k == 0 or self.vllm_top_k < -1:
            raise ValueError("vllm_top_k must be -1 or a positive integer")
        if not 0 <= self.vllm_min_p <= 1:
            raise ValueError("vllm_min_p must be in [0, 1]")
        if not 0 < self.vllm_gpu_memory_utilization <= 1:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if (
            min(self.max_length, self.max_prompt_length, self.max_completion_length)
            <= 0
        ):
            raise ValueError("Sequence lengths must be positive")
        if self.max_length <= self.max_prompt_length:
            raise ValueError("max_length must be greater than max_prompt_length")
        if min(self.per_device_train_batch_size, self.gradient_accumulation_steps) <= 0:
            raise ValueError("Batch size and gradient accumulation must be positive")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not self.reward_funcs:
            raise ValueError("reward_funcs cannot be empty")
        if not 0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        if not 0 <= self.distillation_weight <= 1:
            raise ValueError("distillation_weight must be in [0, 1]")
        if self.reward_weights and len(self.reward_weights) != len(self.reward_funcs):
            raise ValueError("reward_weights must match reward_funcs")
        if self.checkpointing_steps not in {None, "epoch"}:
            if int(self.checkpointing_steps) <= 0:
                raise ValueError(
                    "checkpointing_steps must be a positive integer or 'epoch'"
                )
        if self.push_to_hub and not self.hf_repo_id:
            raise ValueError("push_to_hub requires an explicit hf_repo_id")
