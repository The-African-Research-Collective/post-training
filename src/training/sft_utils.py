import torch

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