"""
Some Training Notes:

local_main_process typically refers to the main process on a specific node/machine in a distributed setup. There can be multiple local_main_processes - one per node.
main_process (or global main process) refers to the single primary process that coordinates across all nodes in the distributed system. There is only one main_process across the entire training setup.
"""

import os
import time
import logging
import datasets
import transformers
import torch
import deepspeed
import functools
import random
import math
import json

from datetime import timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Union
from accelerate.utils import InitProcessGroupKwargs, set_seed, DataLoaderConfiguration
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    DataCollatorForSeq2Seq,
    OPTForCausalLM,
    get_scheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    ArgumentParserPlus,
    mix_datasets,
    CHAT_TEMPLATES,
    get_last_checkpoint_path,
    clean_last_n_checkpoints,
    upload_metadata_to_hf,
    push_folder_to_hub,
)
from model_utils import save_with_accelerate
from src.training.sft_args import ExperimentArguments, ModelArguments, DatasetArguments

logger = get_logger(__name__)


def check_token_ids_in_text(text_ids, token_ids):
    results = []
    for i in range(len(text_ids) - len(token_ids) + 1):
        if text_ids[i : i + len(token_ids)] == token_ids:
            results.append((i, i + len(token_ids)))
    return results


def encode_sft_example(example, tokenizer, max_seq_length, mask_instructions=True):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )

    labels = input_ids.clone()

    list_labels = labels.flatten().tolist()
    modify_labels = False
    tokens_to_mask = [
        "<|start_header_id|>user<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
    ]
    list_of_tokens_ids_to_mask = [
        tokenizer(t, add_special_tokens=False).input_ids for t in tokens_to_mask
    ]

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[
                        :message_idx
                    ],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]

            # set the label to -100 for the non-assistant part
            # The reason why we set this value to -100 is so that the loss is ignored for these tokens.
            # This is because when calculating cross-entropy using pytorch, the function provides a 'ignore_index' parameter
            # that allows us to ignore certain tokens when calculating the loss and the default value for this parameter is -100.
            # From Torch Documentation "" ignore_index (int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient.""
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

            if mask_instructions:
                print(labels[:, message_start_idx:message_end_idx])
                labels[:, message_start_idx:message_end_idx] = -100
                print(labels)
                print(labels[:, message_start_idx:message_end_idx])
            else:
                # Set special tokens to -100
                # TODO: This is very hacky, we should find a better way to do this
                # Also this only works for the llama chat template
                modify_labels = True

                for i, token_ids in enumerate(list_of_tokens_ids_to_mask):
                    results = check_token_ids_in_text(list_labels, token_ids)
                    for j, (x, y) in enumerate(results):
                        list_labels[x:y] = [-100] * (y - x)

                        if i == 0 and j == 0:
                            results.append((0, x))

            if max_seq_length and message_end_idx >= max_seq_length:
                break

    if modify_labels:
        labels = torch.tensor(list_labels).reshape(labels.shape)

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def main(args: ArgumentParserPlus):
    exp_args, model_args, data_args = args[0], args[1], args[2]

    exp_args.output_dir = os.path.join(exp_args.output_dir, exp_args.exp_name)

    exp_args.run_name = f"{exp_args.exp_name}__{exp_args.seed}__{int(time.time())}"
    if exp_args.push_to_hub:
        exp_args.run_name = f"{exp_args.run_name}__hub"

        if exp_args.hf_entity is None:
            exp_args.hf_entity = "taresco"
        if exp_args.hf_repo_id is None:
            exp_args.hf_repo_id = f"{exp_args.hf_entity}/{exp_args.exp_name}"
        if exp_args.hf_repo_revision is None:
            exp_args.hf_repo_revision = exp_args.run_name

        exp_args.hf_repo_url = f"https://huggingface.co/{exp_args.hf_repo_id}/tree/{exp_args.hf_repo_revision}"

    accelerator_log_kwargs = {}

    if exp_args.with_tracking:
        accelerator_log_kwargs["log_with"] = exp_args.report_to
        accelerator_log_kwargs["project_dir"] = exp_args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=exp_args.timeout))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    accelerator = Accelerator(
        gradient_accumulation_steps=exp_args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if exp_args.seed:
        set_seed(exp_args.seed)

    # Create output directory in the main process
    if accelerator.is_main_process:
        if exp_args.output_dir is not None:
            os.makedirs(exp_args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if data_args.dataset_name:
        dataset = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    elif data_args.dataset_mixer:
        dataset = mix_datasets(
            data_args.dataset_mixer,
            configs=data_args.dataset_config_name,
            splits=["train"],
            save_data_dir=data_args.dataset_mix_dir
            if accelerator.is_main_process
            else None,
            columns_to_keep=["messages"],
        )
    elif data_args.dataset_mixer_list:
        dataset = mix_datasets(
            data_args.dataset_mixer_list,
            configs=data_args.dataset_config_name,
            splits=["train"],
            save_data_dir=data_args.dataset_mix_dir
            if accelerator.is_main_process
            else None,
            columns_to_keep=["messages"],
        )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        dataset = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # load pretrained model and tokenizer
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You need to specify either a model name or a model configuration name"
        )

    # load tokenizer
    tokenizer_revision = (
        model_args.tokenizer_revision
        if model_args.tokenizer_revision
        else model_args.model_revision
    )
    if tokenizer_revision != model_args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{model_args.model_revision}`."""
        logger.warning(warning)

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            revision=tokenizer_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=not model_args.use_slow_tokenizer,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=tokenizer_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=not model_args.use_slow_tokenizer,
        )
    else:
        raise ValueError("You need to specify either a tokenizer name or a model name")

    if model_args.model_name_or_path:
        if model_args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=model_args.trust_remote_code,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
                if exp_args.use_flash_attention
                else "eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=model_args.trust_remote_code,
                attn_implementation="flash_attention_2"
                if exp_args.use_flash_attention
                else "eager",
            )
    else:
        logger.info("Training from scratch, no weights or quantization needed")
        model = AutoModelForCausalLM.from_config(config)

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(
        tokenizer, LlamaTokenizerFast
    ):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert (
            num_added_tokens
            in [
                0,
                1,
            ]
        ), "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert model_args.add_bos_token, "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )
            assert (
                num_added_tokens == 1
            ), "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif (
        isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        and tokenizer.pad_token is None
    ):
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert (
            num_added_tokens == 1
        ), "We detected no padding token but add_special_tokens did not add one."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if data_args.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[data_args.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(
                data_args.chat_template_name
            ).chat_template
        except Exception:
            raise ValueError(
                f"Could not find chat template for {data_args.chat_template_name}."
            )

    if model_args.add_bos_token:
        if tokenizer.chat_template.startswith("{{ bos_token }}") or (
            tokenizer.add_bos_token is not None
            and tokenizer.chat_template.startswith(tokenizer.bos_token)
        ):
            raise ValueError(
                "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
            )
        # also add bos in the chat template if not already there
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    if model_args.use_lora:
        if model_args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=model_args.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif model_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = dataset["train"]
    # debugging tool for fewer samples
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        logger.info(
            f"Limiting training samples to {max_train_samples} from {len(train_dataset)}."
        )
        train_dataset = train_dataset.select(range(max_train_samples))

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            functools.partial(
                encode_sft_example,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                mask_instructions=exp_args.mask_instructions,
            ),
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=[
                name
                for name in train_dataset.column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        train_dataset.set_format(type="pt")
        # remove examples with no user messages
        train_dataset = train_dataset.filter(
            lambda example: (example["labels"] != -100).any()
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
        batch_size=exp_args.per_device_train_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": exp_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if model_args.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=exp_args.learning_rate,
            optim_bits=8 if exp_args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=exp_args.learning_rate,
            fused=exp_args.fused_optimizer,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / exp_args.gradient_accumulation_steps
    )
    if exp_args.max_train_steps is None:
        exp_args.max_train_steps = (
            exp_args.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        exp_args.max_train_steps
        if overrode_max_train_steps
        else exp_args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=exp_args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * exp_args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / exp_args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        exp_args.max_train_steps = (
            exp_args.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    exp_args.num_train_epochs = math.ceil(
        exp_args.max_train_steps / num_update_steps_per_epoch
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = exp_args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if exp_args.with_tracking:
        experiment_config = vars(exp_args)
        dataset_args = vars(data_args)
        modelling_args = vars(model_args)

        all_config = {**experiment_config, **dataset_args, **modelling_args}

        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        if exp_args.wandb_entity is None:
            raise ValueError("Please provide a wandb entity.")

        accelerator.init_trackers(
            exp_args.wandb_project_name,
            all_config,
            init_kwargs={
                "wandb": {
                    "name": exp_args.run_name,
                    "entity": exp_args.wandb_entity,
                    "tags": [exp_args.exp_name],
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")

    # Train!
    total_batch_size = (
        exp_args.per_device_train_batch_size
        * accelerator.num_processes
        * exp_args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {exp_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {exp_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {exp_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {exp_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(exp_args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(exp_args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * exp_args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // exp_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    print(f"Starting from epoch {starting_epoch} and step {completed_steps}.")
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    total_token_including_padding = torch.tensor(
        0, dtype=torch.int64, device=accelerator.device
    )
    # TODO: Count multilingual tokens

    start_time = time.time()

    for epoch in range(starting_epoch, exp_args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0

        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            local_total_tokens += batch["attention_mask"].sum()
            total_token_including_padding += batch["attention_mask"].numel()
            with accelerator.accumulate(model):
                if exp_args.load_balancing_loss:
                    outputs = model(**batch, use_cache=False, output_router_logits=True)
                else:
                    outputs = model(**batch, use_cache=False)
                if exp_args.reduce_loss == "mean":
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    if exp_args.load_balancing_loss:
                        aux_loss = exp_args.load_balancing_weight * outputs.aux_loss
                        loss += aux_loss

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if exp_args.load_balancing_loss:
                    total_aux_loss += aux_loss.detach().float()

                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and exp_args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), exp_args.clip_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if (
                    exp_args.logging_steps
                    and completed_steps % exp_args.logging_steps == 0
                ):
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / exp_args.gradient_accumulation_steps
                        / exp_args.logging_steps
                    )
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_tokens_including_padding = (
                        accelerator.gather(total_token_including_padding).sum().item()
                    )
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "total_tokens": total_tokens,
                        "per_device_tps": total_tokens
                        / accelerator.num_processes
                        / (time.time() - start_time),
                        "total_tokens_including_padding": total_tokens_including_padding,
                        "per_device_tps_including_padding": total_tokens_including_padding
                        / accelerator.num_processes
                        / (time.time() - start_time),
                    }
                    if exp_args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / exp_args.gradient_accumulation_steps
                            / exp_args.logging_steps
                        )
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )
                    if exp_args.with_tracking:
                        accelerator.log(
                            metrics_to_log,
                            step=completed_steps,
                        )
                    total_loss = 0
                    total_aux_loss = 0

                if completed_steps >= exp_args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if exp_args.output_dir is not None:
                output_dir = os.path.join(exp_args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(
                os.path.join(
                    get_last_checkpoint_path(exp_args, incomplete=True), "COMPLETED"
                ),
                "w",
            ) as f:
                f.write(
                    "COMPLETED"
                )  # annoyingly, empty files arent uploaded by beaker.
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(
                    exp_args.output_dir, exp_args.keep_last_n_checkpoints
                )
            accelerator.wait_for_everyone()

    if exp_args.output_dir is not None:
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            exp_args.output_dir,
            model_args.use_lora,
        )

    # remove all checkpoints to save space
    if accelerator.is_local_main_process:
        clean_last_n_checkpoints(exp_args.output_dir, keep_last_n_checkpoints=0)

    if accelerator.is_main_process:
        # dpo script only supports these two options right now for datasets
        if data_args.dataset_mixer:
            dataset_list = list(data_args.dataset_mixer.keys())
        elif data_args.dataset_mixer_list:
            dataset_list = data_args.dataset_mixer_list[::2]  # even indices
        elif data_args.dataset_name:
            dataset_list = [data_args.dataset_name]
        else:
            dataset_list = [data_args.train_file]

        # mainly just focussing here on what would be useful for the leaderboard.
        # wandb will have even more useful information.
        metadata_blob = {
            "model_name": exp_args.exp_name,
            "model_type": "sft",
            "datasets": dataset_list,
            "base_model": model_args.model_name_or_path,
            "wandb_path": wandb_tracker.run.get_url(),
        }
        # save metadata to the output directory. then it should also get pushed to HF.
        with open(os.path.join(exp_args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata_blob, f)

        # # upload metadata to the dataset if set
        # if exp_args.hf_metadata_dataset:
        #     upload_metadata_to_hf(
        #         metadata_blob,
        #         "metadata.json",
        #         exp_args.hf_metadata_dataset,
        #         "results/" + exp_args.run_name,  # to match what the auto-evals name as.
        #     )

    if exp_args.push_to_hub:
        push_folder_to_hub(
            accelerator,
            exp_args.output_dir,
            exp_args.hf_repo_id,
            exp_args.hf_repo_revision,
        )

    accelerator.wait_for_everyone()
    if exp_args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParserPlus((ExperimentArguments, ModelArguments, DatasetArguments))
    args = parser.parse()
    main(args)
