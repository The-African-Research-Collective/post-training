from collections import OrderedDict
from typing import Optional

import torch
from accelerate import Accelerator
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer


def save_with_accelerate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    use_lora: bool = False,
    model_attribute_to_save: Optional[str] = None,
) -> None:
    """Save a portable model, or one named model attribute, with Accelerate."""
    # Clear sampling settings so model serialization does not reject them.
    unwrapped_model: PreTrainedModel = accelerator.unwrap_model(model)
    unwrapped_model.generation_config = GenerationConfig(
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    if model_attribute_to_save is not None:
        unwrapped_model = getattr(unwrapped_model, model_attribute_to_save)
    # The wrapped model is required to assemble a complete distributed state dict.
    state_dict = accelerator.get_state_dict(model)

    # A selected submodule needs its prefix removed from the assembled keys.
    if model_attribute_to_save is not None and accelerator.is_main_process:
        state_dict = OrderedDict(
            {
                k[len(f"{model_attribute_to_save}.") :]: v
                for k, v in state_dict.items()
                if k.startswith(f"{model_attribute_to_save}.")
            }
        )

    if use_lora:
        # PeftModel saves adapters only and does not accept is_main_process.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
