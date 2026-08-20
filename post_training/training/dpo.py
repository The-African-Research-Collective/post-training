# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Low-level Direct Preference Optimization (DPO).

The preference objective and concatenated chosen/rejected forward pass are
adapted from TRL revision 6297c47772df3ebb5eef48c3347f75465949256f. The
training loop uses Accelerate directly so every important operation is visible.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from post_training.training.alignment_args import AlignmentArguments
from post_training.training.alignment_utils import (
    create_accelerator,
    create_optimizer,
    create_scheduler,
    encode_preference,
    finish_training,
    load_policy_model,
    load_reference_model,
    load_tokenizer,
    load_train_dataset,
    mean_metrics,
    pad_sequences,
    parse_alignment_args,
    resume_training,
    save_checkpoint,
    selective_log_softmax,
    training_steps,
)


logger = logging.getLogger(__name__)


class PreferenceCollator:
    """Tokenize and concatenate chosen/rejected examples in one model batch."""

    def __init__(self, tokenizer, args: AlignmentArguments):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, examples):
        required = {
            self.args.prompt_column,
            self.args.chosen_column,
            self.args.rejected_column,
        }
        for index, example in enumerate(examples):
            missing = required - example.keys()
            if missing:
                raise ValueError(f"Dataset row {index} is missing columns: {missing}")

        sequences = []
        completion_masks = []
        for completion_column in (
            self.args.chosen_column,
            self.args.rejected_column,
        ):
            for example in examples:
                prompt_ids, completion_ids = encode_preference(
                    self.tokenizer,
                    self.args,
                    example[self.args.prompt_column],
                    example[completion_column],
                )
                sequences.append([*prompt_ids, *completion_ids])
                completion_masks.append(
                    [0] * len(prompt_ids) + [1] * len(completion_ids)
                )

        input_ids = pad_sequences(sequences, self.tokenizer.pad_token_id)
        attention_mask = pad_sequences(
            [[1] * len(sequence) for sequence in sequences], 0
        )
        completion_mask = pad_sequences(completion_masks, 0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "pair_count": len(examples),
        }


def sequence_logps(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum completion-token log probabilities for every concatenated sequence."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    token_logps = selective_log_softmax(
        outputs.logits[:, :-1], batch["input_ids"][:, 1:]
    )
    shifted_mask = batch["completion_mask"][:, 1:]
    return (token_logps * shifted_mask).sum(dim=1)


def dpo_objective(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    pair_count: int,
    args: AlignmentArguments,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer chosen completions relative to the frozen reference policy."""
    policy_chosen, policy_rejected = policy_logps.split(pair_count)
    reference_chosen, reference_rejected = reference_logps.split(pair_count)

    chosen_log_ratio = policy_chosen - reference_chosen
    rejected_log_ratio = policy_rejected - reference_rejected
    logits = args.beta * (chosen_log_ratio - rejected_log_ratio)
    losses = -(1 - args.label_smoothing) * F.logsigmoid(logits)
    losses -= args.label_smoothing * F.logsigmoid(-logits)

    chosen_reward = args.beta * chosen_log_ratio.detach()
    rejected_reward = args.beta * rejected_log_ratio.detach()
    metrics = {
        "loss": losses.mean().detach(),
        "chosen_reward": chosen_reward.mean(),
        "rejected_reward": rejected_reward.mean(),
        "reward_accuracy": (chosen_reward > rejected_reward).float().mean(),
        "reward_margin": (chosen_reward - rejected_reward).mean(),
    }
    return losses.mean(), metrics


def train(args: AlignmentArguments) -> None:
    if args.beta == 0:
        raise ValueError("DPO requires beta greater than zero")
    accelerator = create_accelerator(args)
    tokenizer = load_tokenizer(args)
    dataset = load_train_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=PreferenceCollator(tokenizer, args),
    )

    policy = load_policy_model(args)
    reference = load_reference_model(args)
    optimizer = create_optimizer(policy, args)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
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
        f"DPO: {len(dataset)} examples, {total_steps} optimizer steps, "
        f"{accelerator.num_processes} process(es)"
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
        for batch in epoch_dataloader:
            with accelerator.accumulate(policy):
                policy_logps = sequence_logps(policy, batch)
                with torch.no_grad():
                    reference_logps = sequence_logps(reference, batch)
                loss, metrics = dpo_objective(
                    policy_logps,
                    reference_logps,
                    int(batch["pair_count"]),
                    args,
                )
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
        algorithm="dpo",
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
