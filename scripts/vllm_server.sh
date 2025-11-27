#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_DIR="/tmp"
LOG_DIR="${VLLM_LOG_DIR:-${REPO_ROOT}/logs/vllm}"

mkdir -p "${LOG_DIR}"

print_usage() {
    cat <<EOF
Usage:
  $0 start --port <int> --gpus <ids> [options] [-- vllm-extra-args]
  $0 stop  --port <int>
  $0 status --port <int>

Options (for start):
  --model NAME              HuggingFace repo or local path (default: Qwen/Qwen3-32B)
  --max-model-len INT       Max context length (default: 32768)
  --tensor-parallel INT     Tensor parallel size (default: # of GPUs provided)
  --gpu-mem FLOAT           GPU memory utilization (default: 0.9)
  --metrics-port INT        Optional Prometheus metrics port
  --disable-prefix-cache    Disable vLLM prefix caching
  --disable-eager           Disable enforce-eager flag
  --log-prefix NAME         Override log file prefix (default: vllm_<port>)

Examples:
  $0 start --port 8000 --gpus 0,1 --model Qwen/Qwen3-32B
  $0 stop  --port 8000
EOF
}

require_arg() {
    local name="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "Missing required argument: ${name}" >&2
        exit 1
    fi
}

start_server() {
    local port=""
    local gpus=""
    local model="Qwen/Qwen3-32B"
    local max_len="32768"
    local tensor_parallel=""
    local gpu_mem="0.9"
    local metrics_port=""
    local prefix_cache="true"
    local enforce_eager="true"
    local log_prefix=""
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            --gpus) gpus="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --max-model-len) max_len="$2"; shift 2 ;;
            --tensor-parallel) tensor_parallel="$2"; shift 2 ;;
            --gpu-mem) gpu_mem="$2"; shift 2 ;;
            --metrics-port) metrics_port="$2"; shift 2 ;;
            --disable-prefix-cache) prefix_cache="false"; shift ;;
            --disable-eager) enforce_eager="false"; shift ;;
            --log-prefix) log_prefix="$2"; shift 2 ;;
            --) shift; extra_args=("$@"); break ;;
            *) echo "Unknown option: $1" >&2; print_usage; exit 1 ;;
        esac
    done

    require_arg "--port" "${port}"
    require_arg "--gpus" "${gpus}"

    IFS=',' read -r -a gpu_array <<< "${gpus}"
    if [[ -z "${tensor_parallel}" ]]; then
        tensor_parallel="${#gpu_array[@]}"
    fi

    local pid_file="${PID_DIR}/vllm_${port}.pid"
    if [[ -f "${pid_file}" ]]; then
        echo "PID file ${pid_file} already exists. Stop the server first or remove the PID file." >&2
        exit 1
    fi

    local log_file="${LOG_DIR}/${log_prefix:-vllm_${port}}.log"
    echo "Starting vLLM on port ${port} (GPUs ${gpus})..."
    echo "Logs: ${log_file}"

    local cmd=(vllm serve "${model}"
        --host 0.0.0.0
        --port "${port}"
        --max-model-len "${max_len}"
        --tensor-parallel-size "${tensor_parallel}"
        --gpu-memory-utilization "${gpu_mem}"
    )

    if [[ -n "${metrics_port}" ]]; then
        cmd+=(--metrics-port "${metrics_port}")
    fi
    if [[ "${prefix_cache}" == "true" ]]; then
        cmd+=(--enable-prefix-caching)
    fi
    if [[ "${enforce_eager}" == "true" ]]; then
        cmd+=(--enforce-eager)
    fi
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    CUDA_VISIBLE_DEVICES="${gpus}" nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    local server_pid=$!
    echo "${server_pid}" > "${pid_file}"
    echo "vLLM started with PID ${server_pid}"
}

stop_server() {
    local port=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            *) echo "Unknown option: $1" >&2; print_usage; exit 1 ;;
        esac
    done
    require_arg "--port" "${port}"
    local pid_file="${PID_DIR}/vllm_${port}.pid"
    if [[ ! -f "${pid_file}" ]]; then
        echo "No PID file found at ${pid_file}. Is the server running?" >&2
        exit 1
    fi
    local pid
    pid="$(cat "${pid_file}")"
    if kill "${pid}" >/dev/null 2>&1; then
        echo "Sent SIGTERM to PID ${pid}."
    else
        echo "Failed to send SIGTERM to PID ${pid} (maybe already stopped)." >&2
    fi
    rm -f "${pid_file}"
}

status_server() {
    local port=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            *) echo "Unknown option: $1" >&2; print_usage; exit 1 ;;
        esac
    done
    require_arg "--port" "${port}"
    local pid_file="${PID_DIR}/vllm_${port}.pid"
    if [[ ! -f "${pid_file}" ]]; then
        echo "No server recorded for port ${port}."
        exit 1
    fi
    local pid
    pid="$(cat "${pid_file}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
        echo "vLLM on port ${port} is running with PID ${pid}."
    else
        echo "PID ${pid} from ${pid_file} is not running."
        exit 1
    fi
}

main() {
    if [[ $# -lt 1 ]]; then
        print_usage
        exit 1
    fi
    local action="$1"
    shift
    case "${action}" in
        start) start_server "$@" ;;
        stop) stop_server "$@" ;;
        status) status_server "$@" ;;
        *) print_usage; exit 1 ;;
    esac
}

main "$@"

