import torch
import transformers
from collections import OrderedDict

from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional

from accelerate import Accelerator


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
    model.generation_config = transformers.GenerationConfig(
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
