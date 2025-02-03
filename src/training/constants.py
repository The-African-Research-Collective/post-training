# Preference dataset
# NOTE: the `INPUT_IDS_PROMPT_KEY` is just for visualization purposes only
# also we don't really need `CHOSEN_ATTENTION_MASK_KEY` and `REJECTED_ATTENTION_MASK_KEY`
# since we are always padding from the right with a collator; however they might become
# more useful if we want to do some sort of packing in the future. The nice thing is
# that the tokenization logic would work for both DPO and RM training.
DEFAULT_CHOSEN_KEY = "chosen"
DEFAULT_REJECTED_KEY = "rejected"
CHOSEN_INPUT_IDS_KEY = "chosen_input_ids"
CHOSEN_ATTENTION_MASK_KEY = "chosen_attention_mask"
CHOSEN_LABELS_KEY = "chosen_labels"
REJECTED_INPUT_IDS_KEY = "rejected_input_ids"
REJECTED_ATTENTION_MASK_KEY = "rejected_attention_mask"
REJECTED_LABELS_KEY = "rejected_labels"

INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"
GROUND_TRUTHS_KEY = "ground_truth"
DATASET_SOURCE_KEY = "dataset"

TOKENIZED_PREFERENCE_DATASET_KEYS = [
    CHOSEN_INPUT_IDS_KEY,
    CHOSEN_LABELS_KEY,
    CHOSEN_ATTENTION_MASK_KEY,
    REJECTED_INPUT_IDS_KEY,
    REJECTED_LABELS_KEY,
    REJECTED_ATTENTION_MASK_KEY,
]