"""Run the shared low-level SFT trainer on Modal GPUs."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

import modal


APP_NAME = "post-training-sft"
CACHE_PATH = "/cache"
OUTPUT_PATH = "/outputs"
MAX_TIMEOUT_SECONDS = 24 * 60 * 60

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(
    "post-training-cache", create_if_missing=True
)
output_volume = modal.Volume.from_name(
    "post-training-outputs", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git")
    .uv_sync(
        uv_project_dir=".",
        frozen=True,
        extras=["gpu", "tracking"],
    )
    .env(
        {
            "HF_HOME": f"{CACHE_PATH}/huggingface",
            "HF_DATASETS_CACHE": f"{CACHE_PATH}/huggingface/datasets",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_HOME": f"{CACHE_PATH}/torch",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("post_training")
)


@app.function(
    image=image,
    volumes={CACHE_PATH: cache_volume, OUTPUT_PATH: output_volume},
    timeout=MAX_TIMEOUT_SECONDS,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
)
def train(
    config_text: str,
    *,
    num_gpus: int,
    precision: str,
    overrides: list[str],
    deepspeed_config_text: str | None,
) -> None:
    """Materialize local config text and launch the canonical trainer."""
    with tempfile.TemporaryDirectory(prefix="post-training-sft-") as temp_dir:
        config_path = Path(temp_dir) / "sft.yaml"
        config_path.write_text(config_text)

        command = [
            "accelerate",
            "launch",
            "--mixed_precision",
            precision,
            "--num_machines",
            "1",
            "--num_processes",
            str(num_gpus),
            "--main_process_port",
            "29501",
        ]

        if deepspeed_config_text is not None:
            deepspeed_path = Path(temp_dir) / "deepspeed.json"
            deepspeed_path.write_text(deepspeed_config_text)
            command.extend(
                [
                    "--use_deepspeed",
                    "--deepspeed_config_file",
                    str(deepspeed_path),
                ]
            )
        elif num_gpus > 1:
            command.append("--multi_gpu")

        command.extend(
            [
                "-m",
                "post_training.training.sft",
                str(config_path),
                *overrides,
                f"--output_dir={OUTPUT_PATH}",
            ]
        )
        subprocess.run(command, check=True)

    output_volume.commit()
    cache_volume.commit()


def _read_optional_file(path: str) -> str | None:
    return Path(path).read_text() if path else None


@app.local_entrypoint()
def main(
    config: str,
    gpu: str = "A100-80GB",
    num_gpus: int = 1,
    precision: str = "bf16",
    timeout: int = MAX_TIMEOUT_SECONDS,
    secret: str = "",
    deepspeed_config: str = "",
    overrides: str = "",
) -> None:
    """Submit one resumable SFT run from a local YAML configuration."""
    if num_gpus < 1 or num_gpus > 8:
        raise ValueError("num_gpus must be between 1 and 8")
    if precision not in {"no", "fp16", "bf16", "fp8"}:
        raise ValueError("precision must be one of: no, fp16, bf16, fp8")
    if timeout < 10 or timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout must be between 10 and 86400 seconds")

    parsed_overrides = shlex.split(overrides)
    invalid_overrides = [
        value
        for value in parsed_overrides
        if not value.startswith("--") or "=" not in value
    ]
    if invalid_overrides:
        raise ValueError(
            "Overrides must use --name=value syntax: "
            + ", ".join(invalid_overrides)
        )

    gpu_request = gpu if num_gpus == 1 else f"{gpu}:{num_gpus}"
    secrets = [modal.Secret.from_name(secret)] if secret else []
    configured_train = train.with_options(
        gpu=gpu_request,
        timeout=timeout,
        secrets=secrets,
    )
    configured_train.spawn(
        Path(config).read_text(),
        num_gpus=num_gpus,
        precision=precision,
        overrides=parsed_overrides,
        deepspeed_config_text=_read_optional_file(deepspeed_config),
    ).get()
