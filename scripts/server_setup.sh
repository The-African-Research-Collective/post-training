#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen --extra gpu --extra tracking

# These CUDA-dependent GRPO additions remain explicit because they require the
# target server's toolchain and a research-specific TRL revision.
uv pip install flash-attn --no-build-isolation
uv pip install "trl[vllm] @ git+https://github.com/huggingface/trl.git@1bca49515ecd5b85d16e68c42c76670e252e19f1"

