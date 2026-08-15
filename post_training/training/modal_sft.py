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
CUDA_IMAGE = "nvidia/cuda:12.4.0-devel-ubuntu22.04"
SUPPORTED_DISTRIBUTED_BACKENDS = {"auto", "ddp", "deepspeed", "fsdp"}

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(
    "post-training-cache", create_if_missing=True
)
output_volume = modal.Volume.from_name(
    "post-training-outputs", create_if_missing=True
)


def _resolve_distributed_backend(
    backend: str, *, num_gpus: int, has_deepspeed_config: bool
) -> str:
    """Resolve and validate the requested Accelerate distributed backend."""
    if backend not in SUPPORTED_DISTRIBUTED_BACKENDS:
        raise ValueError(
            "distributed_backend must be one of: auto, ddp, deepspeed, fsdp"
        )
    if backend == "auto":
        if has_deepspeed_config:
            return "deepspeed"
        return "ddp" if num_gpus > 1 else "single"
    if backend == "deepspeed" and not has_deepspeed_config:
        raise ValueError("deepspeed requires --deepspeed-config")
    if backend != "deepspeed" and has_deepspeed_config:
        raise ValueError(
            "--deepspeed-config can only be used with auto or deepspeed"
        )
    if backend in {"ddp", "fsdp"} and num_gpus < 2:
        raise ValueError(f"{backend} requires at least two GPUs")
    return backend


def _distributed_launch_args(
    backend: str,
    *,
    deepspeed_config_path: Path | None = None,
    fsdp_sharding_strategy: str = "FULL_SHARD",
) -> list[str]:
    """Build the explicit Accelerate arguments for one distributed backend."""
    if backend == "deepspeed":
        return [
            "--use_deepspeed",
            "--deepspeed_config_file",
            str(deepspeed_config_path),
        ]
    if backend == "fsdp":
        return [
            "--use_fsdp",
            "--fsdp_sharding_strategy",
            fsdp_sharding_strategy,
            "--fsdp_auto_wrap_policy",
            "TRANSFORMER_BASED_WRAP",
            "--fsdp_state_dict_type",
            "FULL_STATE_DICT",
            "--fsdp_use_orig_params",
            "true",
            "--fsdp_cpu_ram_efficient_loading",
            "true",
            "--fsdp_sync_module_states",
            "true",
        ]
    if backend == "ddp":
        return ["--multi_gpu"]
    return []


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .entrypoint([])
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
            "TRACKIO_DIR": f"{CACHE_PATH}/trackio",
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
    distributed_backend: str,
    fsdp_sharding_strategy: str,
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

        backend = _resolve_distributed_backend(
            distributed_backend,
            num_gpus=num_gpus,
            has_deepspeed_config=deepspeed_config_text is not None,
        )
        deepspeed_path = None
        if deepspeed_config_text is not None:
            deepspeed_path = Path(temp_dir) / "deepspeed.json"
            deepspeed_path.write_text(deepspeed_config_text)
        command.extend(
            _distributed_launch_args(
                backend,
                deepspeed_config_path=deepspeed_path,
                fsdp_sharding_strategy=fsdp_sharding_strategy,
            )
        )

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
    gpu: str = "A10G",
    num_gpus: int = 1,
    precision: str = "bf16",
    timeout: int = MAX_TIMEOUT_SECONDS,
    secret: str = "",
    deepspeed_config: str = "",
    distributed_backend: str = "auto",
    fsdp_sharding_strategy: str = "FULL_SHARD",
    overrides: str = "",
) -> None:
    """Submit one resumable SFT run from a local YAML configuration."""
    if num_gpus < 1 or num_gpus > 8:
        raise ValueError("num_gpus must be between 1 and 8")
    if precision not in {"no", "fp16", "bf16", "fp8"}:
        raise ValueError("precision must be one of: no, fp16, bf16, fp8")
    if timeout < 10 or timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout must be between 10 and 86400 seconds")
    if fsdp_sharding_strategy not in {
        "FULL_SHARD",
        "SHARD_GRAD_OP",
        "HYBRID_SHARD",
        "HYBRID_SHARD_ZERO2",
    }:
        raise ValueError(
            "fsdp_sharding_strategy must be FULL_SHARD, SHARD_GRAD_OP, "
            "HYBRID_SHARD, or HYBRID_SHARD_ZERO2"
        )

    deepspeed_config_text = _read_optional_file(deepspeed_config)
    _resolve_distributed_backend(
        distributed_backend,
        num_gpus=num_gpus,
        has_deepspeed_config=deepspeed_config_text is not None,
    )

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
        deepspeed_config_text=deepspeed_config_text,
        distributed_backend=distributed_backend,
        fsdp_sharding_strategy=fsdp_sharding_strategy,
    ).get()
