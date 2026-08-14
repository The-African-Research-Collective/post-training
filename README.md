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
  --gpu A100-80GB \
  --num-gpus 1 \
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
  --gpu A100-80GB \
  --num-gpus 1 \
  --secret post-training \
  --overrides="--dataset_name=taresco/challenging_math_10k_samples_gpt4_generated --max_train_samples=32 --num_train_epochs=1 --push_to_hub=false --with_tracking=true --report_to=trackio --trackio_project_name=post-training --trackio_space_id=<owner>/<space> --exp_name=challenging_math_modal_smoke --run_name=challenging_math_modal_smoke"
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

## Other workflows

Data generation, translation, evaluation, DPO, and GRPO code remains available
under `post_training/`. These paths are being hardened incrementally; SFT is the
first workflow moved onto the shared, dataset-extensible runtime contract.
