# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""vLLM rollout sampling shared by the low-level GRPO and SDPO loops.

The server protocol, NCCL weight synchronization, DDP prompt gathering, and
PEFT name normalization follow TRL 0.17's VLLMClient and GRPOTrainer. vLLM runs
on separate visible GPUs; the training process only sends policy weights and
receives generated token IDs.
"""

from __future__ import annotations

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model

from post_training.training.alignment_args import AlignmentArguments
from post_training.training.alignment_utils import pad_sequences


class VLLMRolloutSampler:
    """Keep one vLLM server synchronized with a single-GPU or DDP policy."""

    def __init__(self, args: AlignmentArguments, accelerator: Accelerator) -> None:
        if args.tokenizer_name not in {None, args.model_name_or_path}:
            raise ValueError(
                "vLLM variants require the model tokenizer; tokenizer_name must "
                "be unset or match model_name_or_path"
            )
        if args.tokenizer_revision not in {None, args.model_revision}:
            raise ValueError(
                "vLLM variants require tokenizer_revision to match model_revision"
            )
        if args.trust_remote_code:
            raise ValueError(
                "The TRL 0.17 vLLM server does not expose trust_remote_code; "
                "use a model that does not require remote code"
            )

        self.args = args
        self.accelerator = accelerator
        self.client = None
        self._loaded_policy_version: int | None = None

        if accelerator.is_main_process:
            try:
                from trl.extras.vllm_client import VLLMClient
            except (ImportError, ModuleNotFoundError) as error:
                raise ImportError(
                    "grpo_vllm and sdpo_vllm require the Linux-only vllm extra: "
                    "uv sync --frozen --extra gpu --extra vllm"
                ) from error

            self.client = VLLMClient(
                host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                connection_timeout=args.vllm_server_timeout,
            )
            self.client.init_communicator()

        accelerator.wait_for_everyone()

    @staticmethod
    def _vllm_parameter_name(model, name: str) -> str | None:
        """Map a merged PEFT parameter back to the base model's vLLM name."""
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        if model.prefix in name or "original_module" in name:
            return None
        return name.replace("modules_to_save.default.", "")

    def _sync_policy(self, model, policy_version: int) -> None:
        """Send weights once per optimizer version, not per accumulation batch."""
        if policy_version == self._loaded_policy_version:
            return

        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped):
                unwrapped.merge_adapter()
                try:
                    for name, parameter in unwrapped.named_parameters():
                        vllm_name = self._vllm_parameter_name(unwrapped, name)
                        if vllm_name is not None:
                            self.client.update_named_param(vllm_name, parameter.data)
                finally:
                    unwrapped.unmerge_adapter()
            else:
                self.client.update_model_params(unwrapped)
            self.client.reset_prefix_cache()

        self.accelerator.wait_for_everyone()
        self._loaded_policy_version = policy_version

    def generate(
        self,
        model,
        tokenizer,
        base_prompts: list[str],
        policy_version: int,
    ):
        """Generate all ranks' completion groups in one server request."""
        full_prompt_ids = tokenizer(
            base_prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        too_long = [
            (self.accelerator.process_index, index, len(token_ids))
            for index, token_ids in enumerate(full_prompt_ids)
            if len(token_ids) > self.args.max_prompt_length
        ]
        all_too_long = gather_object(too_long)
        if all_too_long:
            raise ValueError(
                "vLLM rollout prompts must already fit max_prompt_length so "
                "generation and policy scoring use identical context; overlong "
                f"(rank, row, tokens): {all_too_long}"
            )
        self._sync_policy(model, policy_version)

        local_prompt_counts = gather_object([len(base_prompts)])
        all_prompts = gather_object(base_prompts)
        total_completions = len(all_prompts) * self.args.num_generations

        generation_error = None
        if self.accelerator.is_main_process:
            try:
                completion_ids = self.client.generate(
                    prompts=all_prompts,
                    n=self.args.num_generations,
                    repetition_penalty=self.args.vllm_repetition_penalty,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.vllm_top_k,
                    min_p=self.args.vllm_min_p,
                    max_tokens=self.args.max_completion_length,
                )
                if len(completion_ids) != total_completions:
                    generation_error = (
                        f"vLLM returned {len(completion_ids)} completions; "
                        f"expected {total_completions}"
                    )
            except Exception as error:
                completion_ids = []
                generation_error = f"{type(error).__name__}: {error}"
        else:
            completion_ids = None

        completion_ids, generation_error = broadcast_object_list(
            [completion_ids, generation_error], from_process=0
        )
        if generation_error is not None:
            raise RuntimeError(f"vLLM generation failed: {generation_error}")
        local_start = (
            sum(local_prompt_counts[: self.accelerator.process_index])
            * self.args.num_generations
        )
        local_size = len(base_prompts) * self.args.num_generations
        local_completion_ids = completion_ids[local_start : local_start + local_size]
        empty_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.pad_token_id
        )
        local_completion_ids = [
            token_ids if token_ids else [empty_token_id]
            for token_ids in local_completion_ids
        ]
        completion_mask = pad_sequences(
            [[1] * len(token_ids) for token_ids in local_completion_ids], 0
        ).to(self.accelerator.device)
        completion_ids = pad_sequences(local_completion_ids, tokenizer.pad_token_id).to(
            self.accelerator.device
        )
        return completion_ids, completion_mask

    def close(self) -> None:
        """Release the temporary NCCL weight-update group."""
        if self.accelerator.is_main_process and self.client is not None:
            self.client.close_communicator()
            self.client = None
