#!/usr/bin/env bash
# Stop vLLM servers on ports 8000-8005.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORTS=(8000 8002 8004)

for port in "${PORTS[@]}"; do
    echo "Stopping port ${port}..."
    if ! "${SCRIPT_DIR}/vllm_server.sh" stop --port "${port}"; then
        echo "  Port ${port}: not running or already stopped."
    fi
done

