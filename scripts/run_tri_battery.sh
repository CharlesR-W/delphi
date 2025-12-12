#!/bin/bash
# Run three experiment battery subsets in parallel on different ports

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/tri_battery"

mkdir -p "$LOG_DIR"

# Edit these lists to choose experiments and servers.
# All experiments are divided round-robin across the listed servers.
EXPERIMENTS=(
    bestofk_baseline
    bestofk_oneshot
    bestofk_train40
    bestofk_embedding_prefilter
    bestofk_random-baseline_from-bestofk_baseline
    iterative_baseline
    iterative_rounds10
    iterative_carry-last
    iterative_always-new-train
    iterative_no-tp-tn
    iterative_history-only
    iterative_train40
)

# Servers/ports that already have vLLM running (edit as needed)
SERVERS=(8000 8002 8004)

# Parse common args
MAX_LATENTS=200
PLOT_ONLY=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-latents)
            MAX_LATENTS="$2"
            shift 2
            ;;
        --plot-only)
            PLOT_ONLY=true
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--max-latents N] [--plot-only] [-- extra args...]"
            exit 1
            ;;
    esac
done

# Build command args
CMD_ARGS=("--max-latents" "$MAX_LATENTS")
if [ "$PLOT_ONLY" = true ]; then
    CMD_ARGS+=("--plot-only")
fi
CMD_ARGS+=("${EXTRA_ARGS[@]}")

# Function to run a group
run_group() {
    local group_num=$1
    local port=$2
    shift 2
    local experiments=("$@")
    
    local log_file="$LOG_DIR/group${group_num}.log"
    local pid_file="$LOG_DIR/group${group_num}.pid"
    
    echo "Starting group $group_num on port $port..."
    echo "  Experiments: ${experiments[*]}"
    echo "  Log: $log_file"
    
    cd "$REPO_ROOT"
    # Resolve Python: env override > repo venv > conda env > uv > PATH
    local python_cmd="${RUN_PYTHON:-}"
    if [[ -z "$python_cmd" ]]; then
        if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
            python_cmd="$REPO_ROOT/.venv/bin/python"
        elif [[ -x "$HOME/miniconda3/envs/iterative/bin/python" ]]; then
            python_cmd="$HOME/miniconda3/envs/iterative/bin/python"
        elif command -v conda >/dev/null 2>&1; then
            python_cmd="conda run -n iterative python"
        elif command -v uv >/dev/null 2>&1; then
            python_cmd="uv run python"
        else
            python_cmd="python"
        fi
    fi
    
    nohup $python_cmd "$SCRIPT_DIR/run_battery_subset.py" \
        --server-port "$port" \
        --experiments "${experiments[@]}" \
        "${CMD_ARGS[@]}" \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "$pid" > "$pid_file"
    echo "  PID: $pid"
    echo "  Monitor: tail -f $log_file"
    echo "  Stop: kill $pid"
    echo
}

# Check if servers are running
check_server() {
    local port=$1
    if ! curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
        echo "WARNING: Server on port $port may not be running!"
        echo "  Start with: $SCRIPT_DIR/vllm_server.sh start --port $port --gpus <GPUS>"
    fi
}

# Round-robin assign experiments to servers
assignments=()
for _ in "${SERVERS[@]}"; do
    assignments+=("")
done

idx=0
for exp in "${EXPERIMENTS[@]}"; do
    server_idx=$((idx % ${#SERVERS[@]}))
    assignments[$server_idx]="${assignments[$server_idx]} ${exp}"
    idx=$((idx + 1))
done

echo "=========================================="
echo "Starting experiment batches"
echo "=========================================="
echo "Max latents: $MAX_LATENTS"
echo "Plot only: $PLOT_ONLY"
echo "Servers: ${SERVERS[*]}"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "=========================================="
echo

# Check servers
for port in "${SERVERS[@]}"; do
    check_server "$port"
done
echo

# Start all groups
for i in "${!SERVERS[@]}"; do
    # Trim leading spaces when splitting into array
    read -ra group_exps <<<"${assignments[$i]}"
    if [ ${#group_exps[@]} -eq 0 ]; then
        continue
    fi
    run_group "$((i + 1))" "${SERVERS[$i]}" "${group_exps[@]}"
done

echo "=========================================="
echo "All groups started!"
echo "=========================================="
echo "PIDs saved to: $LOG_DIR/group*.pid"
echo "Logs: $LOG_DIR/group*.log"
echo
echo "To stop all:"
echo "  for pid in $LOG_DIR/group*.pid; do kill \$(cat \$pid) 2>/dev/null || true; done"
echo
echo "To check status:"
echo "  ps -p \$(cat $LOG_DIR/group*.pid | tr '\n' ',' | sed 's/,$//')"

