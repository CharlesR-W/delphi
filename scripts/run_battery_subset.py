#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from delphi.run_experiment import default_run_config, ExperimentRunner
from delphi.experiment_battery import ExperimentBattery, ExperimentDefinition


def _load_requested_names(args: argparse.Namespace) -> list[str]:
    names: set[str] = set()
    if args.experiments:
        names.update(args.experiments)

    if args.experiments_file:
        for line in Path(args.experiments_file).read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            names.add(stripped)

    if not names:
        raise SystemExit("No experiments specified. Use --experiments or --experiments-file.")
    return sorted(names)


def _collect_all_experiments(battery: ExperimentBattery) -> dict[str, ExperimentDefinition]:
    """Collect every experiment definition the battery knows how to build."""

    def _ensure_iterable(candidate: Iterable[ExperimentDefinition] | ExperimentDefinition) -> Iterable[ExperimentDefinition]:
        if isinstance(candidate, ExperimentDefinition):
            return [candidate]
        return candidate

    experiment_map: dict[str, ExperimentDefinition] = {}
    builders = [
        battery.build_bestofk_grid,
        battery.build_bestofk_debug_grid,
        battery.build_random_baseline,
        battery.build_iterative_grid,
    ]

    for builder in builders:
        try:
            built = builder()
        except TypeError:
            # Builder may require args we don't provide; skip it.
            continue

        for experiment in _ensure_iterable(list(built)):
            experiment_map[experiment.name] = experiment

    return experiment_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a subset of ExperimentBattery definitions with dedicated server ports."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Experiment names from experiment_battery.py (space separated).",
    )
    parser.add_argument(
        "--experiments-file",
        type=Path,
        help="Optional text file listing experiment names (one per line).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        required=True,
        help="Port to bind the vLLM server to for this subset.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Optional Prometheus/metrics port for vLLM. Omit to disable.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override results directory (default: repo_root/results).",
    )
    parser.add_argument(
        "--max-latents",
        type=int,
        default=200,
        help="Maximum latents per run (default: 200).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip running experiments and only generate plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiments that would run and exit.",
    )

    args = parser.parse_args()

    requested = _load_requested_names(args)

    base_cfg = default_run_config()
    base_cfg = replace(
        base_cfg,
        max_latents=args.max_latents,
        name=None,
        server_port=args.server_port,
        server_metrics_port=args.metrics_port,
    )

    results_dir = args.results_dir or (Path(__file__).resolve().parent.parent / "results")
    runner = ExperimentRunner(base_results_dir=results_dir)
    battery = ExperimentBattery(runner, base_cfg)

    all_experiments = _collect_all_experiments(battery)
    missing = [name for name in requested if name not in all_experiments]
    if missing:
        raise SystemExit(
            "Unknown experiment(s): "
            + ", ".join(missing)
            + "\nCheck experiment names in experiment_battery.py."
        )

    subset = [all_experiments[name] for name in requested]
    print(f"Running {len(subset)} experiment(s) with server port {args.server_port}:")
    for exp in subset:
        print(f"  - {exp.name}")

    if args.dry_run:
        return

    if not args.plot_only:
        battery.run(subset)

    # Always plot after completion (or to visualize prior runs when --plot-only)
    battery.plot_runs(subset)
    battery.plot_multi_round_runs(subset)


if __name__ == "__main__":
    main()

