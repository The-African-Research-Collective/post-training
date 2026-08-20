#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/run_alignment.sh" grpo "$@"
