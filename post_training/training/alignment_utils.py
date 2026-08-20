# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Small shared pieces for low-level alignment training.

The data layout and log-probability calculations are adapted from TRL at
revision 6297c47772df3ebb5eef48c3347f75465949256f. The surrounding training
loop intentionally uses Accelerate directly instead of Transformers Trainer.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

from post_training.training.alignment_args import AlignmentArguments
from post_training.training.model_utils import save_with_accelerate
from post_training.training.trackio_tracker import TrackioTracker
from post_training.training.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    push_folder_to_hub,
)


logger = logging.getLogger(__name__)
TRL_SOURCE_REVISION = "6297c47772df3ebb5eef48c3347f75465949256f"


def parse_alignment_args() -> AlignmentArguments:
    """Parse one YAML file plus optional ``--name=value`` overrides."""
    parser = ArgumentParserPlus((AlignmentArguments,))
    return parser.parse()


def create_accelerator(args: AlignmentArguments) -> Accelerator:
    """Create the same Accelerate-first runtime used by the SFT workflow."""
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    args.run_name = args.run_name or f"{args.exp_name}__{args.seed}__{int(time.time())}"

    accelerator_kwargs: dict[str, Any] = {"project_dir": args.output_dir}
    if args.with_tracking:
        log_with: list[Any] = [
            target for target in args.report_to if target != "trackio"
        ]
        if "trackio" in args.report_to:
            log_with.append(
                TrackioTracker(
                    project=args.trackio_project_name
                    or args.project_name
                    or args.exp_name,
                    run_name=args.run_name,
                    group=args.exp_name,
                    space_id=args.trackio_space_id,
                )
            )
        accelerator_kwargs["log_with"] = log_with

    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        kwargs_handlers=[timeout],
        **accelerator_kwargs,
    )
    set_seed(args.seed)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.with_tracking:
        config = vars(args).copy()
        config.update(
            distributed_type=accelerator.distributed_type.value,
            num_processes=accelerator.num_processes,
            mixed_precision=accelerator.mixed_precision,
        )
        init_kwargs = {}
        if "wandb" in args.report_to:
            init_kwargs["wandb"] = {"name": args.run_name, "group": args.exp_name}
            if args.wandb_entity:
                init_kwargs["wandb"]["entity"] = args.wandb_entity
        accelerator.init_trackers(
            args.project_name or args.exp_name,
            config=config,
            init_kwargs=init_kwargs,
        )
    return accelerator


def _model_kwargs(
    args: AlignmentArguments,
    model_name: str,
    revision: str | None = None,
) -> dict[str, Any]:
    dtype_by_name = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.torch_dtype not in dtype_by_name:
        raise ValueError(f"Unsupported torch_dtype: {args.torch_dtype}")
    kwargs: dict[str, Any] = {
        "revision": revision or args.model_revision,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype_by_name[args.torch_dtype],
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    logger.info("Loading model %s", model_name)
    return kwargs


def load_tokenizer(args: AlignmentArguments):
    """Load a tokenizer with explicit, generation-safe padding defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        revision=args.tokenizer_revision or args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an eos_token or pad_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_model(args: AlignmentArguments):
    """Load the trainable policy, optionally with a LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **_model_kwargs(args, args.model_name_or_path),
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    if args.use_lora:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
            ),
        )
    disable_dropout(model)
    return model


def load_reference_model(args: AlignmentArguments):
    """Load a frozen reference policy for DPO or KL-regularized online RL."""
    model_name = args.reference_model_name_or_path or args.model_name_or_path
    revision = args.reference_model_revision
    if revision is None:
        revision = "main" if args.reference_model_name_or_path else args.model_revision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_model_kwargs(args, model_name, revision),
    )
    model.config.use_cache = False
    model.requires_grad_(False)
    disable_dropout(model)
    model.eval()
    return model


def disable_dropout(model: torch.nn.Module) -> None:
    """Match TRL's deterministic policy/reference forward behavior."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def load_train_dataset(args: AlignmentArguments) -> Dataset:
    """Load one Hub or JSON/JSONL dataset without algorithm-specific assumptions."""
    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name,
            name=args.dataset_config_name,
            split=args.dataset_train_split,
            revision=args.dataset_revision,
        )
    else:
        dataset = load_dataset("json", data_files=args.train_file, split="train")
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    return dataset


def _as_messages(value: Any, role: str) -> list[dict[str, str]]:
    if isinstance(value, list):
        return value
    return [{"role": role, "content": str(value)}]


def prompt_messages(args: AlignmentArguments, prompt: Any) -> list[dict[str, str]]:
    messages = _as_messages(prompt, "user")
    if args.system_prompt and (not messages or messages[0].get("role") != "system"):
        messages = [{"role": "system", "content": args.system_prompt}, *messages]
    return messages


def render_prompt(tokenizer, args: AlignmentArguments, prompt: Any) -> str:
    """Render plain or conversational prompts into a generation-ready string."""
    if isinstance(prompt, list) or args.system_prompt:
        messages = prompt_messages(args, prompt)
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(message["content"] for message in messages)
    return str(prompt)


def encode_preference(
    tokenizer,
    args: AlignmentArguments,
    prompt: Any,
    completion: Any,
) -> tuple[list[int], list[int]]:
    """Tokenize one prompt/completion pair while preserving the completion mask."""
    if isinstance(prompt, list) or isinstance(completion, list):
        messages = prompt_messages(args, prompt)
        completion_messages = _as_messages(completion, "assistant")
        if not tokenizer.chat_template:
            prompt_text = "\n".join(message["content"] for message in messages)
            completion_text = "\n".join(
                message["content"] for message in completion_messages
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids
            completion_ids = tokenizer(
                completion_text, add_special_tokens=False
            ).input_ids
        else:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                [*messages, *completion_messages],
                tokenize=True,
                add_generation_prompt=False,
            )
            prefix = 0
            for prompt_id, full_id in zip(prompt_ids, full_ids):
                if prompt_id != full_id:
                    break
                prefix += 1
            prompt_ids = full_ids[:prefix]
            completion_ids = full_ids[prefix:]
    else:
        prompt_ids = tokenizer(
            render_prompt(tokenizer, args, prompt), add_special_tokens=True
        ).input_ids
        completion_ids = tokenizer(str(completion), add_special_tokens=False).input_ids

    prompt_ids = prompt_ids[-args.max_prompt_length :]
    capacity = args.max_length - len(prompt_ids)
    if capacity <= 0:
        raise ValueError("max_length must be greater than the retained prompt length")
    completion_ids = completion_ids[:capacity]
    if tokenizer.eos_token_id is not None and (
        not completion_ids or completion_ids[-1] != tokenizer.eos_token_id
    ):
        if len(completion_ids) == capacity:
            completion_ids[-1] = tokenizer.eos_token_id
        else:
            completion_ids.append(tokenizer.eos_token_id)
    if not completion_ids:
        raise ValueError("A preference completion produced no tokens")
    return prompt_ids, completion_ids


def pad_sequences(
    sequences: list[list[int]], padding_value: int, *, left: bool = False
) -> torch.Tensor:
    """Pad integer sequences without hiding where padding is introduced."""
    max_length = max(len(sequence) for sequence in sequences)
    rows = []
    for sequence in sequences:
        padding = [padding_value] * (max_length - len(sequence))
        rows.append([*padding, *sequence] if left else [*sequence, *padding])
    return torch.tensor(rows, dtype=torch.long)


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return only the next-token log probabilities needed by the objectives."""
    shape = labels.shape
    return -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(shape)


def completion_logps(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probabilities over completion tokens only."""
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    prompt_width = prompt_ids.size(1)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, prompt_width - 1 : -1]
    logps = selective_log_softmax(logits, completion_ids)
    return logps * completion_mask


def create_optimizer(model, args: AlignmentArguments):
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def create_scheduler(optimizer, args: AlignmentArguments, total_steps: int):
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    return get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def training_steps(
    args: AlignmentArguments, dataloader: DataLoader
) -> tuple[int, int, int]:
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_train_steps or args.num_train_epochs * updates_per_epoch
    num_epochs = (
        math.ceil(total_steps / updates_per_epoch)
        if args.max_train_steps is not None
        else args.num_train_epochs
    )
    return updates_per_epoch, total_steps, num_epochs


def mean_metrics(
    accelerator: Accelerator, metrics: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Average scalar logging metrics across all training processes."""
    averaged = {}
    for name, value in metrics.items():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=accelerator.device)
        gathered = accelerator.gather_for_metrics(value.detach().reshape(1))
        averaged[name] = gathered.float().mean().item()
    return averaged


def save_checkpoint(
    accelerator: Accelerator,
    args: AlignmentArguments,
    checkpoint_name: str,
) -> None:
    output_dir = os.path.join(args.output_dir, checkpoint_name)
    accelerator.save_state(output_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        Path(output_dir, "COMPLETED").write_text("COMPLETED")
        clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
    accelerator.wait_for_everyone()


def resume_training(
    accelerator: Accelerator,
    args: AlignmentArguments,
    updates_per_epoch: int,
) -> tuple[int, int, int]:
    """Restore state and return completed steps, starting epoch, and skipped batches."""
    checkpoint = get_last_checkpoint_path(args)
    if not checkpoint:
        return 0, 0, 0
    accelerator.load_state(checkpoint)
    name = os.path.basename(checkpoint)
    if name.startswith("epoch_"):
        starting_epoch = int(name.removeprefix("epoch_")) + 1
        completed_steps = starting_epoch * updates_per_epoch
        skipped_batches = 0
    else:
        completed_steps = int(name.removeprefix("step_"))
        starting_epoch = completed_steps // updates_per_epoch
        skipped_batches = (
            completed_steps % updates_per_epoch
        ) * args.gradient_accumulation_steps
    accelerator.print(f"Resumed from checkpoint: {checkpoint}")
    return completed_steps, starting_epoch, skipped_batches


def finish_training(
    accelerator: Accelerator,
    model,
    tokenizer,
    args: AlignmentArguments,
    algorithm: str,
    completed_steps: int,
) -> None:
    """Save a portable final model, run metadata, and optional Hub artifact."""
    save_with_accelerate(
        accelerator,
        model,
        tokenizer,
        args.output_dir,
        use_lora=args.use_lora,
    )
    if accelerator.is_main_process:
        metadata = {
            "algorithm": algorithm,
            "base_model": args.model_name_or_path,
            "model_revision": args.model_revision,
            "dataset": args.dataset_name or args.train_file,
            "dataset_revision": args.dataset_revision,
            "seed": args.seed,
            "completed_steps": completed_steps,
            "distributed_type": accelerator.distributed_type.value,
            "num_processes": accelerator.num_processes,
            "mixed_precision": accelerator.mixed_precision,
            "trl_source_revision": TRL_SOURCE_REVISION,
        }
        Path(args.output_dir, "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
        Path(args.output_dir, "training_config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True)
        )
    accelerator.wait_for_everyone()
    if args.push_to_hub:
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
            private=args.hf_private_repo,
        )
    if args.with_tracking:
        accelerator.end_training()
