"""Dataset loading and normalization for supervised fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from post_training.training.sft_args import DatasetArguments


SUPPORTED_FILE_BUILDERS = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}


def _load_source(
    source: str,
    *,
    config_name: str | None,
    revision: str | None,
    split: str,
) -> Dataset:
    path = Path(source)
    if path.suffix.lower() in SUPPORTED_FILE_BUILDERS:
        return load_dataset(
            SUPPORTED_FILE_BUILDERS[path.suffix.lower()],
            data_files={split: source},
            split=split,
        )
    if path.exists():
        dataset = load_from_disk(source)
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(
                    f"Local dataset {source!r} has no {split!r} split. "
                    f"Available splits: {list(dataset)}"
                )
            return dataset[split]
        return dataset
    return load_dataset(source, config_name, revision=revision, split=split)


def _resolve_format(dataset: Dataset, args: DatasetArguments) -> str:
    if args.dataset_format != "auto":
        return args.dataset_format

    columns = set(dataset.column_names)
    if "input_ids" in columns:
        return "pretokenized"
    if args.messages_column in columns:
        return "conversational"
    if {args.prompt_column, args.completion_column} <= columns:
        return "prompt_completion"
    if args.text_column in columns:
        return "text"
    raise ValueError(
        "Could not infer the SFT dataset format. Expected input_ids, "
        f"{args.messages_column!r}, both {args.prompt_column!r} and "
        f"{args.completion_column!r}, or {args.text_column!r}. "
        "Set dataset_format and the corresponding column arguments explicitly."
    )


def _prompt_completion_to_messages(
    example: Mapping[str, Any], args: DatasetArguments
) -> dict[str, Any]:
    prompt = example[args.prompt_column]
    completion = example[args.completion_column]
    if isinstance(prompt, str) and isinstance(completion, str):
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        }
    if isinstance(prompt, list) and isinstance(completion, list):
        return {"messages": prompt + completion}
    raise ValueError(
        "Prompt and completion must both be strings or both be conversational lists."
    )


def normalize_sft_dataset(
    dataset: Dataset, args: DatasetArguments
) -> tuple[Dataset, str]:
    """Normalize TRL-compatible SFT shapes to stable internal columns."""
    dataset_format = _resolve_format(dataset, args)
    columns = set(dataset.column_names)

    if dataset_format == "conversational":
        if args.messages_column not in columns:
            raise ValueError(
                f"Conversational dataset is missing {args.messages_column!r}."
            )
        if args.messages_column != "messages":
            dataset = dataset.rename_column(args.messages_column, "messages")
    elif dataset_format == "prompt_completion":
        missing = {
            args.prompt_column,
            args.completion_column,
        } - columns
        if missing:
            raise ValueError(f"Prompt/completion dataset is missing columns: {missing}")
        dataset = dataset.map(
            _prompt_completion_to_messages,
            fn_kwargs={"args": args},
            desc="Normalizing prompt/completion data",
        )
        dataset_format = "conversational"
    elif dataset_format == "text":
        if args.text_column not in columns:
            raise ValueError(f"Text dataset is missing {args.text_column!r}.")
        if args.text_column != "text":
            dataset = dataset.rename_column(args.text_column, "text")
    elif dataset_format == "pretokenized":
        if "input_ids" not in columns:
            raise ValueError("Pretokenized dataset is missing 'input_ids'.")

    return dataset, dataset_format


def _filter_languages(dataset: Dataset, args: DatasetArguments) -> Dataset:
    if args.language_subset is None:
        return dataset
    if args.language_column not in dataset.column_names:
        raise ValueError(
            f"Language filtering requested but {args.language_column!r} is absent."
        )
    languages = (
        {args.language_subset}
        if isinstance(args.language_subset, str)
        else set(args.language_subset)
    )
    return dataset.filter(
        lambda example: example[args.language_column] in languages,
        desc=f"Filtering languages: {sorted(languages)}",
    )


def _mixer_mapping(args: DatasetArguments) -> dict[str, int | float]:
    if args.dataset_mixer is not None:
        return dict(args.dataset_mixer)

    items = args.dataset_mixer_list or []
    if len(items) % 2:
        raise ValueError("dataset_mixer_list must contain source/value pairs.")
    mixer = {}
    for source, value in zip(items[::2], items[1::2]):
        number = float(value) if "." in str(value) else int(value)
        mixer[source] = number
    return mixer


def _load_mixture(args: DatasetArguments, seed: int) -> tuple[Dataset, str]:
    mixer = _mixer_mapping(args)
    values = list(mixer.values())
    if not values or any(value <= 0 for value in values):
        raise ValueError("Dataset mixer values must be positive.")

    use_counts = any(value > 1 for value in values)
    if use_counts and not all(isinstance(value, int) for value in values):
        raise ValueError("Dataset mixtures cannot combine sample counts and fractions.")

    configs = [None] * len(mixer)
    if args.dataset_config_name:
        if isinstance(args.dataset_config_name, str):
            configs = [args.dataset_config_name] * len(mixer)
        else:
            configs = list(args.dataset_config_name)
        if len(configs) != len(mixer):
            raise ValueError("Dataset config count must match the mixer source count.")

    subsets = []
    formats = set()
    for (source, amount), config_name in zip(mixer.items(), configs):
        dataset = _load_source(
            source,
            config_name=config_name,
            revision=args.dataset_revision,
            split=args.train_split,
        )
        dataset = _filter_languages(dataset, args)
        dataset, dataset_format = normalize_sft_dataset(dataset, args)
        formats.add(dataset_format)
        dataset = dataset.shuffle(seed=seed)
        sample_count = amount if use_counts else int(len(dataset) * amount)
        if sample_count == 0:
            raise ValueError(
                f"Mixture value {amount} selects zero rows from {source!r}."
            )
        if sample_count > len(dataset):
            raise ValueError(
                f"Requested {sample_count} rows from {source!r}, which has {len(dataset)}."
            )
        subsets.append(dataset.select(range(sample_count)))

    if len(formats) != 1:
        raise ValueError(
            "A dataset mixture must normalize to one common format; got "
            f"{sorted(formats)}. Pre-normalize heterogeneous sources first."
        )
    return concatenate_datasets(subsets).shuffle(seed=seed), formats.pop()


def load_sft_train_dataset(
    args: DatasetArguments, *, seed: int
) -> tuple[Dataset, str]:
    """Load one SFT training split from Hub, local files, disk, or a mixture."""
    if args.dataset_mixer is not None or args.dataset_mixer_list is not None:
        dataset, dataset_format = _load_mixture(args, seed)
    elif args.dataset_name is not None:
        dataset = _load_source(
            args.dataset_name,
            config_name=args.dataset_config_name,
            revision=args.dataset_revision,
            split=args.train_split,
        )
        dataset = _filter_languages(dataset, args)
        dataset, dataset_format = normalize_sft_dataset(dataset, args)
    else:
        dataset = _load_source(
            args.train_file,
            config_name=None,
            revision=None,
            split="train",
        )
        dataset = _filter_languages(dataset, args)
        dataset, dataset_format = normalize_sft_dataset(dataset, args)

    if len(dataset) == 0:
        raise ValueError("The normalized training dataset is empty.")
    return dataset, dataset_format
