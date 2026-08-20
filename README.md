# Post Training

This repository is the shared home for low-level post-training research. It keeps
the training loop inspectable while following the dataset and configuration
conventions used by Hugging Face TRL. African-language adaptation remains an
important research focus, but it is not a constraint on the reusable training
code.

## Supervised fine-tuning

The SFT entry point is `post_training.training.sft`. It accepts one YAML config
or dataclass arguments directly. A YAML value can be overridden with
`--name=value` after the config path.

Supported dataset sources:

- a Hugging Face dataset through `dataset_name`;
- a local JSON, JSONL, CSV, or Parquet file through `train_file`;
- a dataset saved with `datasets.save_to_disk` through `dataset_name`;
- deterministic count- or fraction-based mixtures through `dataset_mixer`.

Supported data shapes mirror the core TRL SFT formats:

- conversational: a `messages` list containing `role` and `content`;
- prompt/completion: string pairs or conversational-list pairs;
- language modelling text in a `text` column;
- pretokenized `input_ids`, with optional `labels` and `attention_mask`.

Use `dataset_format: auto` for inference, or set it explicitly. The corresponding
column names are configurable. `language_subset` is an optional exact-match
filter and accepts arbitrary language labels.

### GPU server

Python 3.10 through 3.12 is supported. Install the core project and the optional
GPU and tracking dependencies:

```bash
uv sync --frozen --extra gpu --extra tracking
```

Launch one or more visible GPUs without DeepSpeed:

```bash
scripts/run_sft.sh configs/models/dummy_sft_lora.yaml 1 bf16
```

Pass a DeepSpeed JSON config as the fourth argument when required:

```bash
scripts/run_sft.sh \
  configs/models/dummy_sft_lora.yaml \
  4 \
  bf16 \
  configs/deep_speed/stage3_offloading_accelerate.conf
```

Select FSDP explicitly for multi-GPU full parameter sharding. The launcher uses
transformer-layer auto wrapping, rank-zero model loading, synchronized module
initialization, original parameters for LoRA compatibility, and a consolidated
full state dict for portable final artifacts:

```bash
DISTRIBUTED_BACKEND=fsdp scripts/run_sft.sh \
  configs/models/dummy_sft_lora.yaml \
  2 \
  bf16
```

`FSDP_SHARDING_STRATEGY` defaults to `FULL_SHARD` and also accepts
`SHARD_GRAD_OP`, `HYBRID_SHARD`, or `HYBRID_SHARD_ZERO2`. QLoRA remains a
DeepSpeed or single-process workflow; FSDP supports full fine-tuning and LoRA
without 4-bit quantization.

`CUDA_VISIBLE_DEVICES` is inherited from the caller. Flash Attention remains an
optional system-level optimization because its installation depends on the CUDA
toolchain; set `use_flash_attention: false` or `attn_implementation: sdpa` when
it is unavailable.

### Modal serverless GPUs

The Modal adapter builds a Python 3.11 image and invokes the same Accelerate SFT
entry point. It does not contain a second trainer. Install and authenticate the
local Modal client:

```bash
uv sync --frozen --extra modal
uv run modal setup
```

For gated Hugging Face models, Trackio Space logging, or W&B tracking, create a
Modal secret containing `HF_TOKEN` and/or `WANDB_API_KEY`, then pass its name
with `--secret`. A Trackio Space requires an `HF_TOKEN` with write permission.

```bash
uv run modal run --detach -m post_training.training.modal_sft \
  --config configs/models/dummy_sft_lora.yaml \
  --gpu A10G \
  --num-gpus 1 \
  --secret post-training \
  --overrides="--use_flash_attention=false --attn_implementation=sdpa"
```

Modal model publication is disabled by default, even when the YAML configuration
sets `push_to_hub: true`. To publish the final model or LoRA adapter, use the
explicit `--push-to-hub` flag and a Modal secret containing a write-enabled
`HF_TOKEN`. Set `hf_repo_id`, `hf_repo_revision`, and `hf_private_repo` in the
YAML file or through overrides. Newly created repositories remain private by
default.

```bash
uv run modal run --detach -m post_training.training.modal_sft \
  --config configs/models/dummy_sft_lora.yaml \
  --gpu A10G \
  --num-gpus 1 \
  --secret post-training \
  --push-to-hub \
  --overrides="--hf_repo_id=<owner>/<repo> --hf_repo_revision=main"
```

For FSDP on Modal, request at least two GPUs and select the backend explicitly:

```bash
uv run modal run --detach -m post_training.training.modal_sft \
  --config configs/models/dummy_sft_lora.yaml \
  --gpu A10G \
  --num-gpus 2 \
  --distributed-backend fsdp \
  --secret post-training \
  --overrides="--use_flash_attention=false --attn_implementation=sdpa"
```

The first run builds the image and can take several minutes. Model and dataset
caches persist in the `post-training-cache` Modal Volume. Checkpoints and final
artifacts persist under `<exp_name>/` in `post-training-outputs`. Runs retry up to
three times and automatically resume from the newest checkpoint carrying a
`COMPLETED` marker. Use a unique `exp_name` for a new experiment. Modal GPU time
is billable; the launcher never runs a job unless explicitly invoked.

### Experiment tracking

Trackio is an optional backend for the same metrics emitted by the low-level SFT
loop. Enable local-first logging with `with_tracking: true` and
`report_to: [trackio]`. Local data is stored under `TRACKIO_DIR` (Trackio's
default cache when unset) and can be viewed with:

```bash
uv sync --frozen --extra tracking
uv run trackio show --project post-training
```

For a Modal run, provide a Hugging Face Space. The local Trackio database is
also persisted at `/cache/trackio` in the `post-training-cache` volume. This
example logs the 32-sample math smoke run to an existing Space; replace
`<owner>/<space>` rather than creating one implicitly:

```bash
uv run modal run --detach -m post_training.training.modal_sft \
  --config configs/models/dummy_sft_lora.yaml \
  --gpu A10G \
  --num-gpus 1 \
  --secret post-training \
  --overrides="--dataset_name=taresco/challenging_math_10k_samples_gpt4_generated --max_train_samples=32 --num_train_epochs=1 --with_tracking=true --report_to=trackio --trackio_project_name=post-training --trackio_space_id=<owner>/<space> --exp_name=challenging_math_modal_smoke --run_name=challenging_math_modal_smoke"
```

Use `--report_to=trackio,wandb` to log to both backends. Trackio is pinned to
the newest release compatible with this repository's Transformers 4.49 stack.

`uv.lock` is the canonical dependency lock. Use `uv sync --frozen` in automated
or remote environments so dependency drift fails fast instead of changing the
environment during a run.

## Artifacts and safety defaults

Local outputs are written to `<output_dir>/<exp_name>/`. Each completed run
contains the model or adapter, tokenizer, `metadata.json`, and retained
`step_<n>` or `epoch_<n>` Accelerator checkpoints. Hub publication is disabled
by default; when enabled, newly created repositories are private by default.
Remote model code is also disabled by default and must be explicitly trusted.

Pin `model_revision` and `dataset_revision` for reproducible research runs. A
dataset or model identifier does not establish permission to use it: review its
card, license, provenance, personal-data handling, and access terms before
training or publishing artifacts.

## Low-level alignment: DPO, GRPO, and SDPO

The alignment entry points use Accelerate and ordinary PyTorch operations rather
than TRL Trainer classes. Their core data layout and objectives are adapted from
Hugging Face TRL at revision
`6297c47772df3ebb5eef48c3347f75465949256f` under Apache-2.0:

- `post_training.training.dpo` performs one concatenated chosen/rejected forward
  pass and the sigmoid DPO objective against a frozen reference model;
- `post_training.training.grpo` generates response groups with PyTorch, evaluates
  configurable local reward functions, normalizes group advantages, and applies
  clipped GRPO with optional reference KL;
- `post_training.training.sdpo` adds TRL's experimental sampled-token SDPO loss.
  The live policy becomes its own teacher when reprompted with another successful
  group response and/or feedback from a configurable dataset column.

The default GRPO and SDPO modules intentionally keep rollout generation in
PyTorch. Separate `grpo_vllm` and `sdpo_vllm` modules use TRL's dedicated vLLM
server protocol for higher-throughput sampling without duplicating either
objective. DPO supports single GPU, DDP, FSDP, and DeepSpeed through the shared
launcher. Both online rollout backends support single GPU and DDP training;
they reject sharded FSDP/DeepSpeed models.
DPO keeps its frozen reference model replicated on each device, so account for
that memory when selecting a model size for FSDP.

### Dataset contracts

DPO requires configurable `prompt`, `chosen`, and `rejected` columns. Values can
be plain strings or conversational `role`/`content` lists. GRPO and SDPO require
a configurable prompt column. The built-in math rewards also use the configured
solution column; other dataset columns are passed to reward functions unchanged.
SDPO optionally reads environment feedback from `feedback_column`.
Custom reward functions can be configured as `package.module:function`; they
receive `completions`, rendered `prompts`, `solution`, and every original dataset
column as keyword arguments and must return one float or `None` per completion.

Start with the small configs in `configs/dpo/smoke.yaml`,
`configs/grpo/dummy.yaml`, and `configs/sdpo/smoke.yaml`, then replace the model,
dataset, column mapping, and limits. Run them on a GPU server with:

```bash
scripts/run_alignment.sh dpo configs/dpo/smoke.yaml 1 bf16
scripts/run_alignment.sh grpo configs/grpo/dummy.yaml 1 bf16
scripts/run_alignment.sh sdpo configs/sdpo/smoke.yaml 1 bf16
```

### vLLM rollout sampling

vLLM is an optional Python 3.10–3.11, Linux x86-64 dependency so ordinary
training environments do not install its serving stack. Version 0.8.2 is the
newest release compatible with this repository's pinned Torch 2.6 and
Transformers 4.49 environment. The other workflows continue to support Python
3.12; Modal's vLLM image uses Python 3.11. Because PyPI no longer carries
vLLM 0.8.2's exact xgrammar 0.1.16 pin, `uv` overrides it with xgrammar 0.1.17,
the immediate patch used by vLLM 0.8.3. Keep this compatibility exception
isolated to trusted rollout jobs.

```bash
uv sync --frozen --extra gpu --extra tracking --extra vllm
```

Run the rollout server on GPU 0. The model, revision, dtype, and context length
must match the training config; the example values match
`configs/grpo/dummy.yaml`. Keep this custom weight-sync server on loopback—it is
not an authenticated public inference endpoint.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m trl.scripts.vllm_serve \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision main \
  --tensor_parallel_size 1 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu_memory_utilization 0.85 \
  --dtype bfloat16 \
  --max_model_len 384
```

In another shell, train on a different visible GPU. The vLLM variants gather
prompts across DDP ranks, return token IDs to each rank, and synchronize full or
merged LoRA policy weights once per optimizer update.

```bash
CUDA_VISIBLE_DEVICES=1 scripts/run_alignment.sh \
  grpo_vllm configs/grpo/dummy.yaml 1 bf16

CUDA_VISIBLE_DEVICES=1 scripts/run_alignment.sh \
  sdpo_vllm configs/sdpo/smoke.yaml 1 bf16
```

Server connection, sampling, and memory settings are configurable through
`vllm_server_host`, `vllm_server_port`, `vllm_group_port`,
`vllm_server_timeout`, `vllm_repetition_penalty`, `vllm_top_k`, `vllm_min_p`,
`vllm_gpu_memory_utilization`, and `vllm_enforce_eager`. A custom tokenizer or a
model requiring `trust_remote_code` is rejected because TRL 0.17's rollout
server cannot safely mirror those trainer settings.

The same config and `--name=value` overrides work on Modal. A10G is the default:

```bash
uv run modal run --detach -m post_training.training.modal_alignment \
  --algorithm grpo \
  --config configs/grpo/dummy.yaml \
  --gpu A10G \
  --num-gpus 1 \
  --overrides="--max_train_samples=32 --exp_name=grpo_modal_smoke"
```

For vLLM sampling, select a vLLM variant and reserve at least one additional
rollout GPU. Here `--num-gpus` counts training GPUs and `--rollout-gpus` counts
dedicated vLLM GPUs; Modal requests their sum and isolates the two processes.

```bash
uv run modal run --detach -m post_training.training.modal_alignment \
  --algorithm grpo_vllm \
  --config configs/grpo/dummy.yaml \
  --gpu A10G \
  --num-gpus 1 \
  --rollout-gpus 1 \
  --secret post-training \
  --overrides="--max_train_samples=32 --exp_name=grpo_vllm_modal_smoke"
```

The Modal adapter starts the vLLM server on loopback, waits for it to become
healthy, runs the same Accelerate trainer, and then stops the server. A one-GPU
trainer plus one rollout GPU therefore allocates two billable A10Gs.

Select `--algorithm dpo` or `--algorithm sdpo` for the other trainers. Tracking
uses the same `with_tracking`, `report_to`, and Trackio/W&B settings as SFT.
Publishing remains opt-in through Modal's `--push-to-hub` flag and a secret with
`HF_TOKEN`.

Local artifacts are stored at `<output_dir>/<exp_name>/`. On Modal that becomes
`/outputs/<exp_name>/` inside the persistent `post-training-outputs` Volume.
Accelerator state is retained in `step_<n>` or `epoch_<n>` directories; the final
portable model or LoRA adapter, tokenizer, and `metadata.json` live directly in
the experiment directory alongside the resolved `training_config.json`.

Data generation, translation, and evaluation remain available under
`post_training/` and continue to be hardened independently.
