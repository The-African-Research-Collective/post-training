"""Run the low-level DPO, GRPO, or SDPO trainers on Modal GPUs."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

import modal

from post_training.training.modal_sft import (
    CACHE_PATH,
    MAX_TIMEOUT_SECONDS,
    OUTPUT_PATH,
    _distributed_launch_args,
    _read_optional_file,
    _resolve_distributed_backend,
    cache_volume,
    image,
    output_volume,
)


ALGORITHMS = {"dpo", "grpo", "sdpo"}
app = modal.App("post-training-alignment")


@app.function(
    image=image,
    volumes={CACHE_PATH: cache_volume, OUTPUT_PATH: output_volume},
    timeout=MAX_TIMEOUT_SECONDS,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
)
def train(
    algorithm: str,
    config_text: str,
    *,
    num_gpus: int,
    precision: str,
    overrides: list[str],
    deepspeed_config_text: str | None,
    distributed_backend: str,
    fsdp_sharding_strategy: str,
) -> None:
    """Materialize one config and launch the selected module with Accelerate."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of: {', '.join(sorted(ALGORITHMS))}")

    backend = _resolve_distributed_backend(
        distributed_backend,
        num_gpus=num_gpus,
        has_deepspeed_config=deepspeed_config_text is not None,
    )
    if algorithm in {"grpo", "sdpo"} and backend in {"deepspeed", "fsdp"}:
        raise ValueError(
            "GRPO/SDPO rollout generation currently supports single GPU or DDP; "
            "use DPO or SFT for FSDP/DeepSpeed runs."
        )

    with tempfile.TemporaryDirectory(prefix=f"post-training-{algorithm}-") as temp_dir:
        config_path = Path(temp_dir) / f"{algorithm}.yaml"
        config_path.write_text(config_text)
        deepspeed_path = None
        if deepspeed_config_text is not None:
            deepspeed_path = Path(temp_dir) / "deepspeed.json"
            deepspeed_path.write_text(deepspeed_config_text)

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
            *_distributed_launch_args(
                backend,
                deepspeed_config_path=deepspeed_path,
                fsdp_sharding_strategy=fsdp_sharding_strategy,
            ),
            "-m",
            f"post_training.training.{algorithm}",
            str(config_path),
            *overrides,
            f"--output_dir={OUTPUT_PATH}",
        ]
        subprocess.run(command, check=True)

    output_volume.commit()
    cache_volume.commit()


@app.local_entrypoint()
def main(
    config: str,
    algorithm: str = "grpo",
    gpu: str = "A10G",
    num_gpus: int = 1,
    precision: str = "bf16",
    timeout: int = MAX_TIMEOUT_SECONDS,
    secret: str = "",
    push_to_hub: bool = False,
    deepspeed_config: str = "",
    distributed_backend: str = "auto",
    fsdp_sharding_strategy: str = "FULL_SHARD",
    overrides: str = "",
) -> None:
    """Submit a resumable DPO, GRPO, or SDPO run from local YAML."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of: {', '.join(sorted(ALGORITHMS))}")
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
        raise ValueError("Unsupported FSDP sharding strategy")
    if push_to_hub and not secret:
        raise ValueError(
            "--push-to-hub requires --secret containing a write-enabled HF_TOKEN"
        )

    deepspeed_config_text = _read_optional_file(deepspeed_config)
    backend = _resolve_distributed_backend(
        distributed_backend,
        num_gpus=num_gpus,
        has_deepspeed_config=deepspeed_config_text is not None,
    )
    if algorithm in {"grpo", "sdpo"} and backend in {"deepspeed", "fsdp"}:
        raise ValueError("GRPO/SDPO support only single GPU or DDP")

    parsed_overrides = shlex.split(overrides)
    invalid = [
        value
        for value in parsed_overrides
        if not value.startswith("--") or "=" not in value
    ]
    if invalid:
        raise ValueError(
            "Overrides must use --name=value syntax: " + ", ".join(invalid)
        )
    if any(value.partition("=")[0] == "--push_to_hub" for value in parsed_overrides):
        raise ValueError(
            "Use the Modal --push-to-hub flag instead of a push_to_hub override"
        )
    parsed_overrides.append(f"--push_to_hub={str(push_to_hub).lower()}")

    gpu_request = gpu if num_gpus == 1 else f"{gpu}:{num_gpus}"
    secrets = [modal.Secret.from_name(secret)] if secret else []
    configured_train = train.with_options(
        gpu=gpu_request,
        timeout=timeout,
        secrets=secrets,
    )
    configured_train.spawn(
        algorithm,
        Path(config).read_text(),
        num_gpus=num_gpus,
        precision=precision,
        overrides=parsed_overrides,
        deepspeed_config_text=deepspeed_config_text,
        distributed_backend=distributed_backend,
        fsdp_sharding_strategy=fsdp_sharding_strategy,
    ).get()
