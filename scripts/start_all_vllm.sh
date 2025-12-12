#!/usr/bin/env bash
# Start vLLM servers on GPUs 0-5 with Qwen3-14B-AWQ.

set -euo pipefail

# Explicitly set compilers to use user's Conda environment to avoid system dependency issues
export CC=/home/charles/miniconda3/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/charles/miniconda3/bin/x86_64-conda-linux-gnu-g++

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
GPU_MEM="${GPU_MEM:-0.8}"
MAX_LEN="${MAX_LEN:-32768}"
LOG_PREFIX_BASE="${LOG_PREFIX_BASE:-vllm}"

PORTS=(8000 8002 8004)
GPUS=("0,1" "2,3" "4,5")

for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    gpu="${GPUS[$i]}"
    log_prefix="${LOG_PREFIX_BASE}_${port}"
    echo "Starting port ${port} on GPU ${gpu}..."
    "${SCRIPT_DIR}/vllm_server.sh" start \
        --port "${port}" \
        --gpus "${gpu}" \
        --model "${MODEL}" \
        --max-model-len "${MAX_LEN}" \
        --gpu-mem "${GPU_MEM}" \
        --tensor-parallel 2 \
        --log-prefix "${log_prefix}"

    # Wait for server to be ready before starting the next one
    echo "Waiting for port ${port} to be ready..."
    timeout=120
    elapsed=0
    # Use python for health check since curl might be missing
    while ! "${REPO_ROOT}/.venv/bin/python" -c "import requests; requests.get('http://localhost:${port}/v1/models', timeout=1).raise_for_status()" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "Timeout waiting for port ${port}!"
            exit 1
        fi
    done
    echo "Port ${port} is ready."
done

