# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Low-level Group Relative Policy Optimization (GRPO).

This readable objective is adapted from TRL revision
6297c47772df3ebb5eef48c3347f75465949256f: sample a group, normalize rewards
within that group, then optimize a clipped policy objective with reference KL.
The default entry point samples with PyTorch; grpo_vllm supplies vLLM tokens to
the same loop.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from post_training.training.alignment_args import AlignmentArguments
from post_training.training.alignment_utils import (
    completion_logps,
    create_accelerator,
    create_optimizer,
    create_scheduler,
    finish_training,
    load_policy_model,
    load_reference_model,
    load_tokenizer,
    load_train_dataset,
    mean_metrics,
    parse_alignment_args,
    resume_training,
    save_checkpoint,
    training_steps,
)
from post_training.training.online_utils import (
    collect_rollouts,
    grpo_objective,
    require_supported_online_runtime,
)


def train(args: AlignmentArguments, *, rollout_backend: str = "pytorch") -> None:
    if rollout_backend not in {"pytorch", "vllm"}:
        raise ValueError("rollout_backend must be pytorch or vllm")
    accelerator = create_accelerator(args)
    require_supported_online_runtime(accelerator)
    tokenizer = load_tokenizer(args)
    dataset = load_train_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: examples,
    )

    policy = load_policy_model(args)
    reference = load_reference_model(args) if args.beta != 0 else None
    optimizer = create_optimizer(policy, args)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    if reference is not None:
        reference = accelerator.prepare_model(
            reference, device_placement=True, evaluation_mode=True
        )

    updates_per_epoch, total_steps, num_epochs = training_steps(args, dataloader)
    scheduler = accelerator.prepare(create_scheduler(optimizer, args, total_steps))
    completed_steps, starting_epoch, skipped_batches = resume_training(
        accelerator, args, updates_per_epoch
    )
    rollout_sampler = None
    if rollout_backend == "vllm":
        from post_training.training.vllm_rollout import VLLMRolloutSampler

        rollout_sampler = VLLMRolloutSampler(args, accelerator)
    checkpointing_steps = (
        int(args.checkpointing_steps)
        if args.checkpointing_steps not in {None, "epoch"}
        else args.checkpointing_steps
    )

    accelerator.print(
        f"GRPO ({rollout_backend} rollouts): {len(dataset)} prompts, "
        f"{args.num_generations} generations/prompt, {total_steps} optimizer steps"
    )
    optimizer.zero_grad()
    stop_training = completed_steps >= total_steps
    for epoch in range(starting_epoch, num_epochs):
        if stop_training:
            break
        epoch_dataloader = dataloader
        if epoch == starting_epoch and skipped_batches:
            epoch_dataloader = accelerator.skip_first_batches(
                dataloader, skipped_batches
            )

        policy.train()
        for examples in epoch_dataloader:
            rollout = collect_rollouts(
                policy,
                reference,
                tokenizer,
                accelerator,
                args,
                examples,
                rollout_sampler=rollout_sampler,
                policy_version=completed_steps,
            )
            with accelerator.accumulate(policy):
                current_logps = completion_logps(
                    policy,
                    rollout.prompt_ids,
                    rollout.prompt_mask,
                    rollout.completion_ids,
                    rollout.completion_mask,
                )
                loss, metrics = grpo_objective(current_logps, rollout, args)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            completed_steps += 1
            metrics["learning_rate"] = torch.tensor(scheduler.get_last_lr()[0])
            for index, reward_name in enumerate(args.reward_funcs):
                metrics[f"reward/{reward_name}"] = torch.nanmean(
                    rollout.reward_components[:, index]
                )
            if completed_steps % args.logging_steps == 0:
                values = mean_metrics(accelerator, metrics)
                accelerator.print(f"step {completed_steps}: {values}")
                if args.with_tracking:
                    accelerator.log(values, step=completed_steps)
            if (
                isinstance(checkpointing_steps, int)
                and completed_steps % checkpointing_steps == 0
            ):
                save_checkpoint(accelerator, args, f"step_{completed_steps}")
            if completed_steps >= total_steps:
                stop_training = True
                break

        if checkpointing_steps == "epoch":
            checkpoint_name = (
                f"epoch_{epoch}"
                if completed_steps % updates_per_epoch == 0
                else f"step_{completed_steps}"
            )
            save_checkpoint(accelerator, args, checkpoint_name)

    if rollout_sampler is not None:
        rollout_sampler.close()
    finish_training(
        accelerator,
        policy,
        tokenizer,
        args,
        algorithm="grpo_vllm" if rollout_backend == "vllm" else "grpo",
        completed_steps=completed_steps,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train(parse_alignment_args())


if __name__ == "__main__":
    main()
