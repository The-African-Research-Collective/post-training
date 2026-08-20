# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Readable rollout and clipped-policy pieces shared by GRPO and SDPO.

The objective follows TRL's GRPO implementation at revision
6297c47772df3ebb5eef48c3347f75465949256f. This module deliberately keeps the
single-policy path: PyTorch generation, grouped rewards, clipped importance
ratios, and optional reference-policy KL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

from post_training.training.alignment_args import AlignmentArguments
from post_training.training.alignment_utils import completion_logps, render_prompt
from post_training.training.grpo_rewards import get_reward_funcs


@dataclass
class RolloutBatch:
    """Everything required to reuse one rollout for an optimizer update."""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_logps: torch.Tensor
    reference_logps: torch.Tensor | None
    advantages: torch.Tensor
    rewards: torch.Tensor
    reward_components: torch.Tensor
    prompt_texts: list[str]
    completion_texts: list[str]
    examples: list[dict[str, Any]]


def require_supported_online_runtime(accelerator: Accelerator) -> None:
    """Keep generation explicit until sharded-model rollout is implemented."""
    supported = {DistributedType.NO, DistributedType.MULTI_GPU}
    if accelerator.distributed_type not in supported:
        raise ValueError(
            "Low-level GRPO/SDPO currently support one GPU or DDP. "
            "FSDP and DeepSpeed need a separate gathered-generation path."
        )


def tokenize_prompts(tokenizer, prompt_texts: list[str], max_length: int):
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        return tokenizer(
            prompt_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side


def _completion_mask(
    completion_ids: torch.Tensor, eos_token_id: int | None
) -> torch.Tensor:
    """Keep generated tokens through the first EOS and ignore later padding."""
    if eos_token_id is None:
        return torch.ones_like(completion_ids)
    is_eos = completion_ids.eq(eos_token_id)
    sequence_length = completion_ids.size(1)
    positions = torch.arange(sequence_length, device=completion_ids.device)
    positions = positions.unsqueeze(0).expand_as(completion_ids)
    first_eos = torch.where(is_eos, positions, sequence_length).min(dim=1).values
    return positions.le(first_eos.unsqueeze(1)).long()


def _evaluate_rewards(
    args: AlignmentArguments,
    repeated_examples: list[dict[str, Any]],
    prompt_texts: list[str],
    completion_texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    reward_functions = get_reward_funcs(args)
    completions = [
        [{"role": "assistant", "content": text}] for text in completion_texts
    ]
    reward_inputs: dict[str, Any] = {
        "completions": completions,
        "prompts": prompt_texts,
        "solution": [
            example.get(args.solution_column, "") for example in repeated_examples
        ],
    }
    for column in repeated_examples[0]:
        if column not in reward_inputs:
            reward_inputs[column] = [example[column] for example in repeated_examples]

    columns = []
    for reward_function in reward_functions:
        values = reward_function(**reward_inputs)
        if len(values) != len(completion_texts):
            raise ValueError(
                f"Reward function {reward_function.__name__} returned "
                f"{len(values)} values for {len(completion_texts)} completions"
            )
        columns.append(
            [float("nan") if value is None else float(value) for value in values]
        )

    components = torch.tensor(columns, dtype=torch.float32, device=device).T
    weights = args.reward_weights or [1.0] * len(reward_functions)
    weighted = components * torch.tensor(weights, device=device).unsqueeze(0)
    valid = torch.isfinite(components)
    if (~valid.any(dim=1)).any():
        raise ValueError("Every completion must receive at least one finite reward")
    rewards = torch.where(valid, weighted, 0.0).sum(dim=1)
    return rewards, components


def collect_rollouts(
    model,
    reference_model,
    tokenizer,
    accelerator: Accelerator,
    args: AlignmentArguments,
    examples: list[dict[str, Any]],
) -> RolloutBatch:
    """Generate a group per prompt, score it, and freeze behavior log-probs."""
    if not examples:
        raise ValueError("Cannot generate a rollout from an empty batch")
    missing = [
        index
        for index, example in enumerate(examples)
        if args.prompt_column not in example
    ]
    if missing:
        raise ValueError(
            f"Dataset is missing prompt column {args.prompt_column!r} "
            f"for batch rows {missing}"
        )

    base_prompts = [
        render_prompt(tokenizer, args, example[args.prompt_column])
        for example in examples
    ]
    encoded = tokenize_prompts(tokenizer, base_prompts, args.max_prompt_length)
    prompt_ids = encoded["input_ids"].to(accelerator.device)
    prompt_mask = encoded["attention_mask"].to(accelerator.device)
    prompt_ids = prompt_ids.repeat_interleave(args.num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(args.num_generations, dim=0)

    repeated_examples = [
        example for example in examples for _ in range(args.num_generations)
    ]
    prompt_texts = [
        prompt for prompt in base_prompts for _ in range(args.num_generations)
    ]

    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    unwrapped.eval()
    with torch.no_grad():
        generated = unwrapped.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if was_training:
        unwrapped.train()

    completion_ids = generated[:, prompt_ids.size(1) :]
    completion_mask = _completion_mask(completion_ids, tokenizer.eos_token_id)
    completion_texts = tokenizer.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    rewards, reward_components = _evaluate_rewards(
        args,
        repeated_examples,
        prompt_texts,
        completion_texts,
        accelerator.device,
    )

    grouped_rewards = rewards.view(-1, args.num_generations)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, keepdim=True, unbiased=False)
    advantages = ((grouped_rewards - group_mean) / (group_std + 1e-4)).flatten()

    with torch.no_grad():
        old_logps = completion_logps(
            unwrapped,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
        )
        reference_logps = None
        if reference_model is not None:
            reference_logps = completion_logps(
                reference_model,
                prompt_ids,
                prompt_mask,
                completion_ids,
                completion_mask,
            )

    return RolloutBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        old_logps=old_logps,
        reference_logps=reference_logps,
        advantages=advantages,
        rewards=rewards,
        reward_components=reward_components,
        prompt_texts=prompt_texts,
        completion_texts=completion_texts,
        examples=repeated_examples,
    )


def grpo_objective(
    current_logps: torch.Tensor,
    rollout: RolloutBatch,
    args: AlignmentArguments,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute TRL's clipped GRPO loss and optional reference-policy KL."""
    log_ratio = current_logps - rollout.old_logps
    ratio = log_ratio.exp()
    unclipped = ratio * rollout.advantages.unsqueeze(1)
    clipped = ratio.clamp(1 - args.epsilon, 1 + args.epsilon)
    clipped = clipped * rollout.advantages.unsqueeze(1)
    per_token_loss = -torch.minimum(unclipped, clipped)

    if rollout.reference_logps is not None and args.beta != 0:
        ref_log_ratio = rollout.reference_logps - current_logps
        per_token_kl = ref_log_ratio.exp() - ref_log_ratio - 1
        per_token_loss = per_token_loss + args.beta * per_token_kl
    else:
        per_token_kl = torch.zeros_like(per_token_loss)

    mask = rollout.completion_mask
    token_count = mask.sum(dim=1).clamp_min(1)
    loss = ((per_token_loss * mask).sum(dim=1) / token_count).mean()
    mean_kl = ((per_token_kl * mask).sum(dim=1) / token_count).mean()
    clip_fraction = (
        ((ratio - 1).abs() > args.epsilon).float() * mask
    ).sum() / mask.sum().clamp_min(1)
    metrics = {
        "loss": loss.detach(),
        "kl": mean_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "reward": rollout.rewards.mean().detach(),
        "reward_std": rollout.rewards.std(unbiased=False).detach(),
    }
    return loss, metrics
