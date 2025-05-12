from pathlib import Path
from typing import Annotated, Optional

import typer
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from typer import Option

from .response_generation import _build_prompt_message
from post_training.llms.utils import get_response_format_for_model

load_dotenv()
app = typer.Typer(pretty_exceptions_enable=True)


def create_task(
    row: dict, model: str, response_schema: dict, max_tokens: int, temperature: float
):
    task = {
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": _build_prompt_message(row),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_schema", "schema": response_schema},
        },
    }
    return task


@app.command()
def create_batch_file(
    model: Annotated[str, Option(help="OpenAI model name")],
    prompt_input_file: Annotated[
        str, Option(help="JSONL file containing input prompts")
    ],
    output_dir: Annotated[str, Option(help="Output directory for batch files created")],
    response_class: Annotated[
        str,
        Option(
            help="Fully-qualified class name: module.class_name. If The module is not a post_training.* submodule, ensure it is placed in the root directory where this script is invoked."
        ),
    ],
    max_tokens: Annotated[int, Option(help="Max tokens for generation")] = 2048,
    temperature: Annotated[float, Option(help="Temperature for generation")] = 0.0,
) -> None:
    input_dataset = load_dataset("json", data_files=prompt_input_file, split="train")

    response_model = get_response_format_for_model(response_class)
    response_schema = response_model.model_json_schema(mode="validation")

    task_dataset = input_dataset.map(
        create_task,
        batched=False,
        remove_columns=input_dataset.column_names,
        fn_kwargs={
            "model": model,
            "response_schema": response_schema,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    batch_dir = Path(output_dir)
    batch_dir.mkdir(exist_ok=True)

    # OpenAI Batch API only permits 50K rows and 200MB files
    # TODO: @theyorubayesian - Handle file size limits
    for idx, batch in enumerate(
        task_dataset.batch(batch_size=50_000, drop_last_batch=False)
    ):
        batch_ds = Dataset.from_generator(
            lambda: (yield from batch), features=batch.features
        )
        batch_ds.to_json(batch_dir / f"{str(idx).zfill(5)}.jsonl", force_acii=False)


@app.command()
def submit_batch_job(
    model: Annotated[str, Option(help="OpenAI model name")],
    response_class: Annotated[
        str,
        Option(
            help="Fully-qualified class name: module.class_name. If The module is not a post_training.* submodule, ensure it is placed in the root directory where this script is invoked."
        ),
    ],
    prompt_input_file: Annotated[
        Optional[str],
        Option(
            help="JSONL file containing input prompts. One of `prompt_input_file` and `batch_input` must be passed."
        ),
    ] = None,
    batch_input: Annotated[
        Optional[str],
        Option(
            help="Batch input file or directory containing already prepared batch input files for submission. One of `prompt_input_file` and `batch_input` must be passed."
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str], Option(help="Output directory for batch files created")
    ] = None,
    max_tokens: Annotated[int, Option(help="Max tokens for generation")] = 2048,
    temperature: Annotated[float, Option(help="Temperature for generation")] = 0.0,
    organization: Annotated[Optional[str], Option(help="OpenAI organization")] = None,
) -> None:
    client = OpenAI(organization=organization)

    if prompt_input_file:
        input_file = Path(prompt_input_file)

        if output_dir is None:
            batch_dir = input_file.parent / input_file.stem + "_task"
            batch_dir.mkdir(exist_ok=True)

        create_batch_file(
            model,
            prompt_input_file,
            response_class=response_class,
            max_tokens=max_tokens,
            temperature=temperature,
            output_dir=batch_dir.as_posix(),
        )
    elif batch_input:
        batch_dir = Path(batch_input)
    else:
        raise ValueError(
            "One of `prompt_input_file` or `batch_input` must be provided."
        )

    if batch_dir.is_file():
        tasks = [batch_dir]
    elif batch_dir.is_dir():
        tasks = list(batch_dir.glob("*.json")) + list(batch_dir.glob("*.jsonl"))
    else:
        raise ValueError("No file or directory found for batch inputs")

    for batch_file in tasks:
        uploaded_file = client.files.create(
            file=batch_file.open(mode="rb"), purpose="batch"
        )
        logger.info(f"Uploaded {batch_file.as_posix()} with id: {uploaded_file.id}")

        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info(f"Created batch job: {batch_job.id}")


@app.command()
def list_batch_jobs(
    organization: Annotated[Optional[str], Option(help="OpenAI organization")] = None,
    after: Annotated[
        Optional[str], Option(help="List all jobs after this job id provided")
    ] = None,
    limit: Annotated[int, Option(help="Number of items per page")] = 20,
) -> None:
    client = OpenAI(organization=organization)

    page = client.batches.list(after=after, limit=limit)

    while page.data is not None:
        for job in page.data:
            logger.info(job.model_dump_json())

        page = client.batches.list(after=page.next_page_info["after"], limit=limit)


if __name__ == "__main__":
    app()
