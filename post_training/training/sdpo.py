# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Low-level Self-Distilled Policy Optimization (SDPO).

This implements TRL's experimental sampled-token SDPO objective at revision
6297c47772df3ebb5eef48c3347f75465949256f. The live policy is also the teacher:
it scores each sampled completion again after the prompt is augmented with a
successful group response and/or environment feedback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
    RolloutBatch,
    collect_rollouts,
    grpo_objective,
    require_supported_online_runtime,
    tokenize_prompts,
)


@dataclass
class TeacherBatch:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    valid_examples: torch.Tensor


def build_teacher_batch(
    rollout: RolloutBatch,
    tokenizer,
    args: AlignmentArguments,
    device: torch.device,
) -> TeacherBatch:
    """Reprompt each rollout with another successful response or feedback."""
    teacher_prompts = []
    valid_examples = []
    group_size = args.num_generations

    for group_start in range(0, len(rollout.examples), group_size):
        group_indices = list(range(group_start, group_start + group_size))
        successful = [
            index
            for index in group_indices
            if rollout.rewards[index].item() >= args.success_reward_threshold
        ]
        for index in group_indices:
            candidates = successful
            if args.dont_reprompt_on_self_success:
                candidates = [
                    candidate for candidate in candidates if candidate != index
                ]

            solution = ""
            if args.use_successful_as_teacher and candidates:
                best = max(
                    candidates,
                    key=lambda candidate: rollout.rewards[candidate].item(),
                )
                solution = args.solution_template.format(
                    successful_previous_attempt=rollout.completion_texts[best]
                )

            feedback = ""
            raw_feedback = rollout.examples[index].get(args.feedback_column)
            if args.include_environment_feedback and raw_feedback:
                feedback = args.feedback_template.format(feedback_raw=raw_feedback)

            valid = bool(solution or feedback)
            valid_examples.append(valid)
            if valid:
                teacher_prompts.append(
                    args.reprompt_template.format(
                        prompt=rollout.prompt_texts[index],
                        solution=solution,
                        feedback=feedback,
                    )
                )
            else:
                teacher_prompts.append(rollout.prompt_texts[index])

    encoded = tokenize_prompts(tokenizer, teacher_prompts, args.max_prompt_length)
    return TeacherBatch(
        prompt_ids=encoded["input_ids"].to(device),
        prompt_mask=encoded["attention_mask"].to(device),
        valid_examples=torch.tensor(valid_examples, device=device, dtype=torch.bool),
    )


def sampled_token_distillation(
    student_logps: torch.Tensor,
    teacher_logps: torch.Tensor,
    rollout: RolloutBatch,
    teacher_batch: TeacherBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate reverse KL on the exact tokens sampled by the student policy."""
    log_ratio = student_logps - teacher_logps
    per_token_loss = log_ratio.detach() * student_logps
    mask = rollout.completion_mask * teacher_batch.valid_examples.unsqueeze(1)
    token_count = mask.sum(dim=1).clamp_min(1)
    per_example = (per_token_loss * mask).sum(dim=1) / token_count
    valid = teacher_batch.valid_examples
    if valid.any():
        loss = per_example[valid].mean()
    else:
        loss = per_example.sum() * 0
    return loss, valid.float().mean()


def train(args: AlignmentArguments) -> None:
    if not args.use_successful_as_teacher and not args.include_environment_feedback:
        raise ValueError(
            "SDPO requires use_successful_as_teacher or include_environment_feedback"
        )

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
    use_policy_loss = args.distillation_weight < 1
    reference = (
        load_reference_model(args) if use_policy_loss and args.beta != 0 else None
    )
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
    checkpointing_steps = (
        int(args.checkpointing_steps)
        if args.checkpointing_steps not in {None, "epoch"}
        else args.checkpointing_steps
    )

    accelerator.print(
        f"SDPO: {len(dataset)} prompts, {args.num_generations} generations/prompt, "
        f"distillation_weight={args.distillation_weight}"
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
                policy, reference, tokenizer, accelerator, args, examples
            )
            teacher_batch = build_teacher_batch(
                rollout, tokenizer, args, accelerator.device
            )
            with accelerator.accumulate(policy):
                student_logps = completion_logps(
                    policy,
                    rollout.prompt_ids,
                    rollout.prompt_mask,
                    rollout.completion_ids,
                    rollout.completion_mask,
                )
                with torch.no_grad():
                    teacher_logps = completion_logps(
                        accelerator.unwrap_model(policy),
                        teacher_batch.prompt_ids,
                        teacher_batch.prompt_mask,
                        rollout.completion_ids,
                        rollout.completion_mask,
                    )
                distillation_loss, teacher_fraction = sampled_token_distillation(
                    student_logps, teacher_logps, rollout, teacher_batch
                )
                policy_loss, metrics = grpo_objective(student_logps, rollout, args)
                loss = (
                    1 - args.distillation_weight
                ) * policy_loss + args.distillation_weight * distillation_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            completed_steps += 1
            metrics.update(
                loss=loss.detach(),
                policy_loss=policy_loss.detach(),
                distillation_loss=distillation_loss.detach(),
                teacher_fraction=teacher_fraction.detach(),
                learning_rate=torch.tensor(scheduler.get_last_lr()[0]),
            )
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

    finish_training(
        accelerator,
        policy,
        tokenizer,
        args,
        algorithm="sdpo",
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
