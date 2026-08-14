from collections import OrderedDict
from typing import Optional

import torch
from accelerate import Accelerator
from trl import ModelConfig, get_kbit_device_map, get_quantization_config
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM

from post_training.training.grpo_args import GRPOConfig


def save_with_accelerate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    use_lora: bool = False,
    model_attribute_to_save: Optional[str] = None,
) -> None:
    """`model_attribute_to_save` is for used to save PPO's policy instead of the full model"""
    # set the generation config to an empty setting to be safe.
    # we usually do greedy decoding for generation, so this should be okay.
    # otherwise, we get an error thrown at save time.
    model.generation_config = GenerationConfig(
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    unwrapped_model: PreTrainedModel = accelerator.unwrap_model(model)
    if model_attribute_to_save is not None:
        unwrapped_model = getattr(unwrapped_model, model_attribute_to_save)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)

    # if we are saving a specific attribute of the model, we need to filter the state_dict
    # also the state_dict only lives in the main process; other processes just have state_dict = None
    if model_attribute_to_save is not None and accelerator.is_main_process:
        state_dict = OrderedDict(
            {
                k[len(f"{model_attribute_to_save}.") :]: v
                for k, v in state_dict.items()
                if k.startswith(f"{model_attribute_to_save}.")
            }
        )

    if use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)


def get_tokenizer(model_args: ModelConfig, training_args: GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: GRPOConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
