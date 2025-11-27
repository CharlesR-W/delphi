#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SUBSET="${SCRIPT_DIR}/run_battery_subset.py"
LOG_DIR="${REPO_ROOT}/logs/tri_battery"

mkdir -p "${LOG_DIR}"

# Define experiment buckets (tweak here if the catalog changes)
GROUP1=(
    bestofk_baseline
    bestofk_oneshot
    bestofk_train40
    bestofk_embedding_prefilter
)
GROUP2=(
    bestofk_random-baseline_from-bestofk_baseline
    iterative_baseline
    iterative_rounds10
    iterative_carry-last
)
GROUP3=(
    iterative_always-new-train
    iterative_no-tp-tn
    iterative_history-only
    iterative_train40
)

PORTS=(8002 8004 8006)
GROUP_NAMES=("group1" "group2" "group3")
GROUPS=(GROUP1 GROUP2 GROUP3)

echo "Launching tri-battery runs (ports ${PORTS[*]}). Extra args: $*"

pids=()
for idx in "${!GROUPS[@]}"; do
    group_var="${GROUPS[$idx]}"
    experiments=("${group_var[@]}") # Indirect expansion not allowed with set -u
    case "${idx}" in
        0) experiments=("${GROUP1[@]}") ;;
        1) experiments=("${GROUP2[@]}") ;;
        2) experiments=("${GROUP3[@]}") ;;
    esac

    port="${PORTS[$idx]}"
    label="${GROUP_NAMES[$idx]}"
    log_file="${LOG_DIR}/${label}.log"

    echo "  * ${label} -> port ${port}, experiments: ${experiments[*]}"

    nohup python "${RUN_SUBSET}" \
        --server-port "${port}" \
        --experiments "${experiments[@]}" \
        "$@" \
        > "${log_file}" 2>&1 &

    pids+=($!)
    echo "    PID ${pids[-1]} (logs: ${log_file})"
done

echo "All tri-battery jobs launched. Active PIDs: ${pids[*]}"


