"""Run a battery of experiments for Best-of-K vs Iterative explainers."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import orjson
import pandas as pd

from delphi.config import RunConfig
from delphi.log.result_analysis import (
    add_latent_f1,
    get_agg_metrics,
    load_data,
    log_results,
)

if __package__ is None or __package__ == "":
    from run_experiment import ExperimentRunner, default_run_config
else:  # pragma: no cover
    from .run_experiment import ExperimentRunner, default_run_config


class ExperimentDefinition:
    def __init__(self, name: str, config: RunConfig) -> None:
        self.name = name
        self.config = config

    def with_name(self, name: str) -> "ExperimentDefinition":
        new_cfg = replace(self.config, name=name)
        return ExperimentDefinition(name, new_cfg)


class ExperimentBattery:
    def __init__(self, runner: ExperimentRunner, base_config: RunConfig) -> None:
        self.runner = runner
        self.base_config = base_config

    def _apply_feedback_caps(self, cfg: RunConfig, include_tp_tn: bool) -> RunConfig:
        if include_tp_tn:
            fp_cap = fn_cap = tp_cap = tn_cap = 5
        else:
            fp_cap = fn_cap = 10
            tp_cap = tn_cap = 0
        return replace(
            cfg,
            iterative_max_num_false_positives=fp_cap,
            iterative_max_num_false_negatives=fn_cap,
            iterative_max_num_true_positives=tp_cap,
            iterative_max_num_true_negatives=tn_cap,
        )

    def _make_iterative_config(
        self,
        *,
        name: str,
        rounds: int,
        carry: str,
        include_tp_tn: bool,
        always_new_train: bool,
        history_only: bool,
    ) -> ExperimentDefinition:
        cfg = replace(
            self.base_config,
            explainer="iterative",
            iterative_num_rounds=rounds,
            explainer_temperature=0.7,
            iterative_carryforward_strategy=carry,
            iterative_history_only=history_only,
            iterative_always_new_train_examples=always_new_train,
            name=name,
        )
        cfg = self._apply_feedback_caps(cfg, include_tp_tn)
        return ExperimentDefinition(name, cfg)

    def build_iterative_grid(self) -> Iterable[ExperimentDefinition]:
        experiments: list[ExperimentDefinition] = []

        experiments.append(
            self._make_iterative_config(
                name="iterative_baseline",
                rounds=5,
                carry="best",
                include_tp_tn=True,
                always_new_train=False,
                history_only=False,
            )
        )

        experiments.append(
            self._make_iterative_config(
                name="iterative_rounds10",
                rounds=10,
                carry="best",
                include_tp_tn=True,
                always_new_train=False,
                history_only=False,
            )
        )

        experiments.append(
            self._make_iterative_config(
                name="iterative_carry-last",
                rounds=5,
                carry="last",
                include_tp_tn=True,
                always_new_train=False,
                history_only=False,
            )
        )

        experiments.append(
            self._make_iterative_config(
                name="iterative_always-new-train",
                rounds=5,
                carry="best",
                include_tp_tn=True,
                always_new_train=True,
                history_only=False,
            )
        )

        experiments.append(
            self._make_iterative_config(
                name="iterative_no-tp-tn",
                rounds=5,
                carry="best",
                include_tp_tn=False,
                always_new_train=False,
                history_only=False,
            )
        )

        experiments.append(
            self._make_iterative_config(
                name="iterative_history-only",
                rounds=5,
                carry="best",
                include_tp_tn=True,
                always_new_train=False,
                history_only=True,
            )
        )

        return experiments

    def build_bestofk_grid(self) -> Iterable[ExperimentDefinition]:
        cfg = replace(
            self.base_config,
            explainer="bestofk",
            bestofk_num_explanations=5,
            explainer_temperature=0.7,
            judge_scorer_index=self.base_config.scorers.index("fuzz")
            if "fuzz" in self.base_config.scorers
            else 1,
            name="bestofk_k10_temp0.7_judge_fuzz",
        )
        yield ExperimentDefinition(cfg.name, cfg)

        # Add multishot variant
        cfg_multishot = replace(
            cfg,
            bestofk_is_multishot=False,
            name="bestofk_k10_temp0.7_judge_fuzz_oneshot",
        )
        yield ExperimentDefinition(cfg_multishot.name, cfg_multishot)

    def build_bestofk_debug_grid(self) -> Iterable[ExperimentDefinition]:
        """Temporary debugging experiments with smaller scale and adjusted parameters"""
        experiments: list[ExperimentDefinition] = []

        # Debug experiment 1: Very small scale with lower min_examples
        cfg_debug1 = replace(
            self.base_config,
            explainer="bestofk",
            bestofk_num_explanations=5,  # Fewer explanations
            explainer_temperature=0.7,
            judge_scorer_index=self.base_config.scorers.index("fuzz")
            if "fuzz" in self.base_config.scorers
            else 0,
            max_latents=10,  # Much smaller
            name="bestofk_debug_k5_temp0.7_judge_fuzz_small",
        )
        # Override constructor config for lower min_examples
        cfg_debug1 = replace(
            cfg_debug1,
            constructor_cfg=replace(
                cfg_debug1.constructor_cfg,
                min_examples=50,  # Lower threshold
            ),
        )
        # Override sampler config for more examples
        cfg_debug1 = replace(
            cfg_debug1,
            sampler_cfg=replace(
                cfg_debug1.sampler_cfg,
                n_examples_train=50,  # More training examples
                n_examples_test=100,  # Keep test examples reasonable
            ),
        )
        # More examples per scorer prompt
        cfg_debug1 = replace(cfg_debug1, num_examples_per_scorer_prompt=10)
        experiments.append(ExperimentDefinition(cfg_debug1.name, cfg_debug1))

        # Debug experiment 2: Even smaller scale
        cfg_debug2 = replace(
            self.base_config,
            explainer="bestofk",
            bestofk_num_explanations=3,  # Even fewer explanations
            explainer_temperature=0.7,
            judge_scorer_index=self.base_config.scorers.index("fuzz")
            if "fuzz" in self.base_config.scorers
            else 0,
            max_latents=5,  # Very small
            name="bestofk_debug_k3_temp0.7_judge_fuzz_tiny",
        )
        # Override constructor config for lower min_examples
        cfg_debug2 = replace(
            cfg_debug2,
            constructor_cfg=replace(
                cfg_debug2.constructor_cfg,
                min_examples=20,  # Very low threshold
            ),
        )
        # Override sampler config for more examples
        cfg_debug2 = replace(
            cfg_debug2,
            sampler_cfg=replace(
                cfg_debug2.sampler_cfg,
                n_examples_train=100,  # More training examples
                n_examples_test=50,  # Fewer test examples for tiny scale
            ),
        )
        # More examples per scorer prompt
        cfg_debug2 = replace(cfg_debug2, num_examples_per_scorer_prompt=15)
        experiments.append(ExperimentDefinition(cfg_debug2.name, cfg_debug2))

        return experiments

    async def run_async(self, experiments: Sequence[ExperimentDefinition]) -> None:
        for exp in experiments:
            await self.runner.run_async(exp.config)

    def run(self, experiments: Sequence[ExperimentDefinition]) -> None:
        asyncio.run(self.run_async(experiments))

    @property
    def results_root(self) -> Path:
        return (self.runner.base_results_dir or Path.cwd() / "results").resolve()

    def _run_dir(self, run_name: str) -> Path:
        return self.results_root / run_name

    def _load_run_config(self, run_name: str) -> dict | None:
        cfg_path = self._run_dir(run_name) / "run_config.json"
        if not cfg_path.exists():
            print(f"[ExperimentBattery] Missing run_config.json for {run_name}")
            return None
        return orjson.loads(cfg_path.read_bytes())

    def plot_runs(self, experiments: Sequence[ExperimentDefinition]) -> None:
        for exp in experiments:
            run_dir = self._run_dir(exp.name)
            cfg_dict = self._load_run_config(exp.name)
            if cfg_dict is None:
                continue
            scores_path = run_dir / "scores"
            visualize_path = run_dir / "visualize"
            if not scores_path.exists():
                print(
                    f"[ExperimentBattery] Scores missing for {exp.name}, skipping plots"
                )
                continue
            log_results(
                scores_path,
                visualize_path,
                cfg_dict.get("hookpoints", []),
                cfg_dict.get("scorers", []),
                image_formats=["png", "pdf"],
            )

    def _collect_metrics(
        self, experiments: Sequence[ExperimentDefinition], scorer: str
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for exp in experiments:
            run_dir = self._run_dir(exp.name)
            cfg_dict = self._load_run_config(exp.name)
            if cfg_dict is None:
                continue
            scores_path = run_dir / "scores"
            if not scores_path.exists():
                continue
            modules = cfg_dict.get("hookpoints", [])
            scorers = cfg_dict.get("scorers", [])
            try:
                latent_df, counts = load_data(scores_path, modules)
            except FileNotFoundError:
                print(f"[ExperimentBattery] Score files missing for {exp.name}")
                continue
            if latent_df is None or latent_df.empty:
                print(f"[ExperimentBattery] No scores loaded for {exp.name}")
                continue
            latent_df = latent_df[latent_df["score_type"].isin(scorers)]
            if latent_df.empty:
                continue
            latent_df = add_latent_f1(latent_df)
            agg_df = get_agg_metrics(latent_df, counts)
            scorer_row = agg_df[agg_df["score_type"] == scorer]
            if scorer_row.empty:
                continue
            top = scorer_row.iloc[0]
            rows.append(
                {
                    "run": exp.name,
                    "explainer": exp.config.explainer,
                    "score_type": scorer,
                    "f1_score": float(top["f1_score"]),
                    "weighted_f1": (
                        float(top["weighted_f1"])
                        if top["weighted_f1"] is not None
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    def plot_cross_experiments(
        self,
        bestofk_experiments: Sequence[ExperimentDefinition],
        iterative_experiments: Sequence[ExperimentDefinition],
        scorers: Sequence[str] = ("fuzz",),
        output_prefix: str = "bestofk_vs_iterative",
    ) -> None:
        import matplotlib.pyplot as plt

        compare_dir = self.results_root / "visualize-cross-experiment"
        compare_dir.mkdir(parents=True, exist_ok=True)

        for scorer in scorers:
            bok_df = self._collect_metrics(bestofk_experiments, scorer)
            it_df = self._collect_metrics(iterative_experiments, scorer)
            combined = pd.concat([bok_df, it_df], ignore_index=True)
            if combined.empty:
                print(f"[ExperimentBattery] No metrics available for scorer '{scorer}'")
                continue
            combined["order"] = range(len(combined))
            for metric in ("f1_score", "weighted_f1"):
                metric_df = combined.dropna(subset=[metric])
                if metric_df.empty:
                    continue

                # Generate both PNG and PDF formats using matplotlib
                for fmt in ("png", "pdf"):
                    output_path = (
                        compare_dir / f"{output_prefix}_{scorer}_{metric}.{fmt}"
                    )

                    fig, ax = plt.subplots(figsize=(14, 8))

                    # Plot each explainer type with different colors/markers
                    for explainer_type in metric_df["explainer"].unique():
                        subset = metric_df[metric_df["explainer"] == explainer_type]
                        marker = "o" if explainer_type == "bestofk" else "s"
                        ax.scatter(
                            subset["order"],
                            subset[metric],
                            label=explainer_type,
                            s=100,
                            alpha=0.7,
                            marker=marker,
                        )

                    ax.set_xlabel("Experiment", fontsize=12)
                    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
                    ax.set_title(
                        f"{metric.replace('_', ' ').title()} comparison ({scorer})",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax.set_xticks(metric_df["order"])
                    ax.set_xticklabels(
                        metric_df["run"], rotation=60, ha="right", fontsize=9
                    )
                    ax.legend(title="Explainer", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    print(
                        f"[ExperimentBattery] Saved cross-experiment plot: {output_path}"
                    )


if __name__ == "__main__":
    base_cfg = default_run_config()

    base_cfg = replace(
        base_cfg,
        max_latents=10,
        # name will be set per experiment
    )

    runner = ExperimentRunner()
    battery = ExperimentBattery(runner, base_cfg)

    # --- ORIGINAL EXPERIMENTS (with improved parameters based on debugging) ---
    bestofk_experiments = list(battery.build_bestofk_grid())
    iterative_experiments = list(battery.build_iterative_grid())

    # --- Commands (uncommented to run) ---
    # Run Best-of-K baseline experiments with FIXED per-round score writing
    print("Running Best-of-K experiments...")
    battery.run(bestofk_experiments)

    # Plot Best-of-K runs
    print("Plotting Best-of-K runs...")
    battery.plot_runs(bestofk_experiments)

    # Run Iterative experiments
    print("Running Iterative experiments...")
    battery.run(iterative_experiments)

    # Plot Iterative runs
    print("Plotting Iterative runs...")
    battery.plot_runs(iterative_experiments)

    # Cross-experiment comparison plots (Best-of-K vs Iterative)
    print("Generating cross-experiment comparison plots...")
    battery.plot_cross_experiments(bestofk_experiments, iterative_experiments)

    # --- Commented-out comparison experiments for detection vs fuzz scorer ---
    # Uncomment to run experiments comparing detection vs fuzz as judge scorer
    """
    def build_scorer_comparison_grid(self) -> Iterable[ExperimentDefinition]:
        experiments: list[ExperimentDefinition] = []

        # Best-of-K with fuzz as judge
        cfg_fuzz = replace(
            self.base_config,
            explainer="bestofk",
            bestofk_num_explanations=10,
            explainer_temperature=0.7,
            judge_scorer_index=self.base_config.scorers.index("fuzz")
            if "fuzz" in self.base_config.scorers
            else 0,
            name="bestofk_k10_temp0.7_judge_fuzz",
        )
        experiments.append(ExperimentDefinition(cfg_fuzz.name, cfg_fuzz))

        # Best-of-K with detection as judge
        cfg_detection = replace(
            self.base_config,
            explainer="bestofk",
            bestofk_num_explanations=10,
            explainer_temperature=0.7,
            judge_scorer_index=self.base_config.scorers.index("detection")
            if "detection" in self.base_config.scorers
            else 1,
            name="bestofk_k10_temp0.7_judge_detection",
        )
        experiments.append(ExperimentDefinition(cfg_detection.name, cfg_detection))

        return experiments

    # Uncomment to run scorer comparison
    # scorer_comparison_experiments = list(battery.build_scorer_comparison_grid())
    # battery.run(scorer_comparison_experiments)
    # battery.plot_runs(scorer_comparison_experiments)
    """
