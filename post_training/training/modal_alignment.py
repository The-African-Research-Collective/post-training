"""Run the low-level DPO, GRPO, or SDPO trainers on Modal GPUs."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import modal
import yaml

from post_training.training.modal_sft import (
    CACHE_PATH,
    MAX_TIMEOUT_SECONDS,
    OUTPUT_PATH,
    _distributed_launch_args,
    _build_image,
    _read_optional_file,
    _resolve_distributed_backend,
    cache_volume,
    image,
    output_volume,
)


ALGORITHMS = {"dpo", "grpo", "grpo_vllm", "sdpo", "sdpo_vllm"}
ONLINE_ALGORITHMS = {"grpo", "grpo_vllm", "sdpo", "sdpo_vllm"}
VLLM_ALGORITHMS = {"grpo_vllm", "sdpo_vllm"}
app = modal.App("post-training-alignment")
vllm_image = _build_image("gpu", "tracking", "vllm")


def _run_trainer(
    algorithm: str,
    config_text: str,
    *,
    num_gpus: int,
    precision: str,
    overrides: list[str],
    deepspeed_config_text: str | None,
    distributed_backend: str,
    fsdp_sharding_strategy: str,
    environment: dict[str, str] | None = None,
) -> None:
    """Materialize one config and launch its low-level Accelerate module."""
    backend = _resolve_distributed_backend(
        distributed_backend,
        num_gpus=num_gpus,
        has_deepspeed_config=deepspeed_config_text is not None,
    )
    if algorithm in ONLINE_ALGORITHMS and backend in {"deepspeed", "fsdp"}:
        raise ValueError(
            "GRPO/SDPO rollout generation supports single GPU or DDP; "
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
        subprocess.run(command, check=True, env=environment)


def _resolved_vllm_settings(
    config_text: str, overrides: list[str]
) -> dict[str, object]:
    """Resolve the small config subset needed before the trainer starts."""
    values = yaml.safe_load(config_text) or {}
    for override in overrides:
        name, _, raw_value = override.removeprefix("--").partition("=")
        values[name] = yaml.safe_load(raw_value)
    tokenizer_name = values.get("tokenizer_name")
    model_name = values.get("model_name_or_path")
    if not model_name:
        raise ValueError("vLLM rollout variants require model_name_or_path")
    if tokenizer_name not in {None, model_name}:
        raise ValueError(
            "vLLM rollout variants require tokenizer_name to match the model"
        )
    if values.get("tokenizer_revision") not in {
        None,
        values.get("model_revision", "main"),
    }:
        raise ValueError(
            "vLLM rollout variants require tokenizer_revision to match the model"
        )
    if values.get("trust_remote_code", False):
        raise ValueError("vLLM rollout variants do not support trust_remote_code")
    server_port = int(values.get("vllm_server_port", 8000))
    group_port = int(values.get("vllm_group_port", 51216))
    if not 0 < server_port <= 65535 or not 0 < group_port <= 65535:
        raise ValueError("vLLM ports must be between 1 and 65535")
    if server_port == group_port:
        raise ValueError("vLLM server and weight-sync ports must differ")
    if float(values.get("vllm_server_timeout", 300.0)) <= 0:
        raise ValueError("vllm_server_timeout must be positive")
    memory_utilization = float(values.get("vllm_gpu_memory_utilization", 0.85))
    if not 0 < memory_utilization <= 1:
        raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")
    return values


def _visible_gpu_ids(total_gpus: int) -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = (
        configured.split(",") if configured else [str(i) for i in range(total_gpus)]
    )
    if len(gpu_ids) < total_gpus:
        raise RuntimeError(
            f"Modal exposed {len(gpu_ids)} GPUs but the run requires {total_gpus}"
        )
    return gpu_ids[:total_gpus]


def _wait_for_vllm(
    process: subprocess.Popen, host: str, port: int, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://{host}:{port}/health/"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM server exited with status {return_code}")
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"vLLM server did not become ready at {health_url}")


def _vllm_server_command(settings: dict[str, object], rollout_gpus: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "trl.scripts.vllm_serve",
        "--model",
        str(settings["model_name_or_path"]),
        "--revision",
        str(settings.get("model_revision", "main")),
        "--tensor_parallel_size",
        str(rollout_gpus),
        "--host",
        str(settings.get("vllm_server_host", "127.0.0.1")),
        "--port",
        str(settings.get("vllm_server_port", 8000)),
        "--gpu_memory_utilization",
        str(settings.get("vllm_gpu_memory_utilization", 0.85)),
        "--dtype",
        str(settings.get("torch_dtype", "bfloat16")),
        "--max_model_len",
        str(
            int(settings.get("max_prompt_length", 512))
            + int(settings.get("max_completion_length", 256))
        ),
    ]
    if settings.get("vllm_enforce_eager", False):
        command.extend(["--enforce_eager", "true"])
    return command


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
    """Launch a trainer that does not need a dedicated rollout GPU."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of: {', '.join(sorted(ALGORITHMS))}")
    if algorithm in VLLM_ALGORITHMS:
        raise ValueError("Use train_vllm for a vLLM rollout algorithm")
    _run_trainer(
        algorithm,
        config_text,
        num_gpus=num_gpus,
        precision=precision,
        overrides=overrides,
        deepspeed_config_text=deepspeed_config_text,
        distributed_backend=distributed_backend,
        fsdp_sharding_strategy=fsdp_sharding_strategy,
    )
    output_volume.commit()
    cache_volume.commit()


@app.function(
    image=vllm_image,
    volumes={CACHE_PATH: cache_volume, OUTPUT_PATH: output_volume},
    timeout=MAX_TIMEOUT_SECONDS,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
)
def train_vllm(
    algorithm: str,
    config_text: str,
    *,
    num_gpus: int,
    rollout_gpus: int,
    precision: str,
    overrides: list[str],
    distributed_backend: str,
    fsdp_sharding_strategy: str,
) -> None:
    """Run a loopback-only vLLM server beside the Accelerate trainer."""
    if algorithm not in VLLM_ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of: {', '.join(sorted(VLLM_ALGORITHMS))}"
        )

    settings = _resolved_vllm_settings(config_text, overrides)
    host = str(settings.get("vllm_server_host", "127.0.0.1"))
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Modal vLLM servers must bind to loopback")
    server_port = int(settings.get("vllm_server_port", 8000))
    server_timeout = float(settings.get("vllm_server_timeout", 300.0))

    gpu_ids = _visible_gpu_ids(num_gpus + rollout_gpus)
    server_environment = os.environ.copy()
    server_environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:rollout_gpus])
    trainer_environment = os.environ.copy()
    trainer_environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[rollout_gpus:])

    server = subprocess.Popen(
        _vllm_server_command(settings, rollout_gpus),
        env=server_environment,
    )
    try:
        _wait_for_vllm(server, host, server_port, server_timeout)
        _run_trainer(
            algorithm,
            config_text,
            num_gpus=num_gpus,
            precision=precision,
            overrides=overrides,
            deepspeed_config_text=None,
            distributed_backend=distributed_backend,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            environment=trainer_environment,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    output_volume.commit()
    cache_volume.commit()


@app.local_entrypoint()
def main(
    config: str,
    algorithm: str = "grpo",
    gpu: str = "A10G",
    num_gpus: int = 1,
    rollout_gpus: int = 1,
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
    if rollout_gpus < 1 or rollout_gpus > 8:
        raise ValueError("rollout_gpus must be between 1 and 8")
    if algorithm in VLLM_ALGORITHMS and num_gpus + rollout_gpus > 8:
        raise ValueError("training and rollout GPUs cannot exceed 8 in total")
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
    if algorithm in ONLINE_ALGORITHMS and backend in {"deepspeed", "fsdp"}:
        raise ValueError("GRPO/SDPO support only single GPU or DDP")
    if algorithm in VLLM_ALGORITHMS and deepspeed_config_text is not None:
        raise ValueError("vLLM rollout variants do not support DeepSpeed")

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

    requested_gpus = (
        num_gpus + rollout_gpus if algorithm in VLLM_ALGORITHMS else num_gpus
    )
    gpu_request = gpu if requested_gpus == 1 else f"{gpu}:{requested_gpus}"
    secrets = [modal.Secret.from_name(secret)] if secret else []
    if algorithm in VLLM_ALGORITHMS:
        configured_train = train_vllm.with_options(
            gpu=gpu_request,
            timeout=timeout,
            secrets=secrets,
        )
        configured_train.spawn(
            algorithm,
            Path(config).read_text(),
            num_gpus=num_gpus,
            rollout_gpus=rollout_gpus,
            precision=precision,
            overrides=parsed_overrides,
            distributed_backend=distributed_backend,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
        ).get()
    else:
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
