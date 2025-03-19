"""
python -m post_training.data.prompt_translation.py \
    taresco/grade_school_math_20k_samples_gpt4_generated data train \
    --batch-size 64 --output-file files/translated_grade_school_math.jsonl
"""

import functools
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from datasets import Dataset, load_dataset
from deep_translator import GoogleTranslator

app = typer.Typer(pretty_exceptions_enable=True)

LANGUAGE_MAP = {
    "AFRIKAANS": "af",
    "ARABIC": "ar",
    "HAUSA": "ha",
    "IGBO": "ig",
    "NIGERIAN PIDGIN": "auto",
    "SOMALI": "so",
    "SWAHILI": "sw",
    "YORUBA": "yo",
    "ZULU": "zu",
}


def explode_messages(examples) -> dict[str, list[str]]:
    exploded_examples = defaultdict(list)

    for index in range(len(examples["id"])):
        for item in examples["messages"][index]:
            for column in ("id", "model", "language"):
                exploded_examples[column].append(examples[column][index])

            exploded_examples["role"].append(item["role"])
            exploded_examples["content"].append(item["content"])

    return exploded_examples


def collapse_messages(examples) -> dict[str, list[str]]:
    collapsed_examples = {
        "id": [examples["id"][0]],
        "model": [examples["model"][0]],
        "language": [examples["language"][0]],
        "messages": [
            [
                {"role": examples["role"][0], "content": examples["content"][0]},
                {"role": examples["role"][1], "content": examples["content"][1]},
            ]
        ],
    }

    return collapsed_examples


def google_translate(
    sentence: str,
    source_lang: str | None = None,
    target_lang: str | None = None,
    translator: GoogleTranslator | None = None,
    proxies: dict[str, str] | None = None,
) -> str | None:
    """
    Translate a sentence from source language to target language
    :param translation_sentence: sentence to be translated
    :param source_lang: source language
    :param target_lang: target language
    :return: translated sentence
    """
    if translator is None:
        assert source_lang is not None and target_lang is not None, (
            "`source_lang` and `target_lang` should not be None if `translator` is None"
        )
        translator = GoogleTranslator(
            source=source_lang, target=target_lang, proxies=proxies
        )

    # TODO: @theyorubayesian - Maybe use a threadpoolexecutor here?
    try:
        translated = translator.translate(text=sentence)
    except Exception:
        translated = None

    return translated


@app.command()
def translate_sft_dataset(
    dataset_name: str,
    dataset_config_name: str,
    dataset_split: str,
    proxylist: str | None = None,
    proxy_server_username: str | None = os.getenv("PROXY_SERVER_USERNAME"),
    proxy_server_password: str | None = os.getenv("PROXY_SERVER_PASSWORD"),
    batch_size: int = 32,
    max_workers: int | None = None,
    output_file: str | None = None,
) -> None:
    proxies = [None]
    if proxylist is not None:
        proxylist = Path(proxylist).read_text().splitlines()
        proxies = [
            {
                "http://": f"{proxy_server_username}:{proxy_server_password}@{proxy}",
                "https://": f"{proxy_server_username}:{proxy_server_password}@{proxy}",
            }
            for proxy in random.sample(proxylist, len(proxylist))
        ]

    dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
    exploded_dataset = dataset.map(
        explode_messages, batched=True, remove_columns=dataset.column_names
    )

    output_dataset = defaultdict(list)

    executor = ThreadPoolExecutor(max_workers=max_workers)

    for batch in exploded_dataset.batch(batch_size=batch_size):
        translations = []
        for key in exploded_dataset.column_names:
            if key == "content":
                futures = [
                    executor.submit(
                        functools.partial(
                            google_translate,
                            source_lang=LANGUAGE_MAP[batch["language"][idx]],
                            target_lang="en",
                            proxies=random.sample(proxies, 1)[0],
                        ),
                        text,
                    )
                    for idx, text in enumerate(batch["content"])
                ]

                for future in futures:
                    translations.append(future.result())

                output_dataset[key].extend(translations)
            else:
                output_dataset[key].extend(batch[key])

    translated_dataset = Dataset.from_dict(output_dataset)
    (
        translated_dataset.map(
            collapse_messages,
            batch_size=2,
            batched=True,
            remove_columns=translated_dataset.column_names,
        ).to_json(output_file)
    )


if __name__ == "__main__":
    app()
