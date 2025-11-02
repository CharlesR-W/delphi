"""Run a battery of experiments for Best-of-K vs Iterative explainers."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
from scipy.stats import gaussian_kde

from delphi.config import RunConfig
from delphi.log.result_analysis import (
    add_latent_f1,
    compute_classification_metrics,
    compute_confusion,
    get_agg_metrics,
    load_data,
    log_results,
)

if __package__ is None or __package__ == "":
    from run_experiment import ExperimentRunner, default_run_config
else:  # pragma: no cover
    from .run_experiment import ExperimentRunner, default_run_config


# ----------------------
# Multi-round explanation analysis helper functions
# ----------------------


def _compute_baseline_metrics(n_pos: int = 100, n_neg: int = 60) -> dict[str, float]:
    """Compute baseline F1 and accuracy metrics for random guessing strategies.
    
    Args:
        n_pos: Number of positive (activating) examples
        n_neg: Number of negative (non-activating) examples
    
    Returns:
        Dictionary with baseline metrics:
        - always_true_f1: F1 score when always predicting positive
        - always_true_accuracy: Accuracy when always predicting positive
        - random_f1: F1 score when predicting positive with p = class frequency
        - random_accuracy: Accuracy when predicting positive with p = class frequency
    """
    total = n_pos + n_neg
    p_pos = n_pos / total
    
    # Always predicting true (positive)
    # TP = n_pos, FP = n_neg, TN = 0, FN = 0
    always_precision = n_pos / (n_pos + n_neg)
    always_recall = 1.0
    always_f1 = 2 * (always_precision * always_recall) / (always_precision + always_recall)
    always_accuracy = n_pos / total
    
    # Random guessing with p = class frequency
    # Expected values:
    exp_tp = n_pos * p_pos
    exp_fp = n_neg * p_pos
    exp_tn = n_neg * (1 - p_pos)
    exp_fn = n_pos * (1 - p_pos)
    
    random_precision = exp_tp / (exp_tp + exp_fp) if (exp_tp + exp_fp) > 0 else 0
    random_recall = exp_tp / (exp_tp + exp_fn) if (exp_tp + exp_fn) > 0 else 0
    random_f1 = 2 * (random_precision * random_recall) / (random_precision + random_recall) if (random_precision + random_recall) > 0 else 0
    random_accuracy = (exp_tp + exp_tn) / total
    
    return {
        "always_true_f1": always_f1,
        "always_true_accuracy": always_accuracy,
        "random_f1": random_f1,
        "random_accuracy": random_accuracy,
    }


def _parse_multi_score_filename(stem: str) -> tuple[str, int, int]:
    """Parse multi-score filename stem into (module, latent_idx, round_idx).

    Expected pattern: "<module>_latent<idx>_<round>"
    Example: "layers.5_latent0_2" -> ("layers.5", 0, 2)
    """
    match = re.match(r"(.+)_latent(\d+)_([0-9]+)$", stem)
    if not match:
        raise ValueError(f"Unexpected multi_scores filename pattern: {stem}")
    module = match.group(1)
    latent_idx = int(match.group(2))
    round_idx = int(match.group(3))
    return module, latent_idx, round_idx


def _load_single_score_file(path: Path) -> pd.DataFrame:
    """Helper to load a single score file to sample-level DataFrame.

    Matches the structure produced by scorer_postprocess (list of dicts per sample).
    """
    try:
        data = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        print(f"Error decoding JSON from {path}. Skipping file.")
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "text": "".join(ex.get("str_tokens", [])),
                "activating": ex.get("activating"),
                "prediction": ex.get("prediction"),
                "probability": ex.get("probability"),
                "correct": ex.get("correct"),
            }
            for ex in data
        ]
    )


def _load_explanation_scores_per_round(scores_path: Path, max_rounds: int | None = None) -> pd.DataFrame:
    """Load per-explanation (per-round) scores from scores/*/multi_scores.

    Returns a DataFrame with columns:
    - module, latent_idx, round, scorer, f1_score, accuracy, precision, recall
    
    Args:
        scores_path: Path to scores directory
        max_rounds: Optional cap on number of rounds/candidates to load per latent.
                   If provided, only loads rounds 0 through (max_rounds-1).
                   Useful for BestOfK to cap at K candidates even if more were generated.
    """
    rows = []
    for scorer_dir in scores_path.iterdir():
        if not scorer_dir.is_dir():
            continue
        multi_dir = scorer_dir / "multi_scores"
        if not multi_dir.exists():
            print(
                f"[load_explanation_scores_per_round] No multi_scores "
                f"found for {scorer_dir.name}, skipping"
            )
            continue

        multi_files = list(multi_dir.glob("*.txt"))
        if len(multi_files) == 0:
            print(
                f"[load_explanation_scores_per_round] multi_scores directory empty "
                f"for {scorer_dir.name}, skipping"
            )
            continue

        for file in multi_files:
            try:
                module, latent_idx, round_idx = _parse_multi_score_filename(file.stem)
            except ValueError as e:
                print(f"[load_explanation_scores_per_round] Skipping {file.name}: {e}")
                continue

            # Apply cap if specified (for BestOfK)
            if max_rounds is not None and round_idx >= max_rounds:
                continue

            df = _load_single_score_file(file)
            if df.empty:
                continue
            conf = compute_confusion(df)
            metrics = compute_classification_metrics(conf)
            rows.append(
                {
                    "module": module,
                    "latent_idx": latent_idx,
                    "round": round_idx,
                    "scorer": scorer_dir.name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def _plot_box_per_round(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """Box-and-whisker plot showing score distribution at each round."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = _compute_baseline_metrics()

    for scorer, sdf in round_df.groupby("scorer"):
        output_path = out_dir / f"{scorer}_bnw_perround.{image_format}"

        fig, ax = plt.subplots(figsize=(12, 6))
        rounds = sorted(sdf["round"].unique())
        data_by_round = [sdf[sdf["round"] == r]["f1_score"].dropna() for r in rounds]

        bp = ax.boxplot(data_by_round, labels=rounds, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        # Add baseline reference lines
        ax.axhline(y=baselines["always_true_f1"], color="gray", linestyle="--", 
                   linewidth=1, alpha=0.5, label="Always guess positive")
        ax.axhline(y=baselines["random_f1"], color="dimgray", linestyle=":", 
                   linewidth=1, alpha=0.5, label="Random (by frequency)")

        # Titles and labels
        fig.suptitle("Frequency-agnostic F1 score distribution by round", fontsize=14, fontweight="bold")
        ax.set_title(run_label, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Frequency-agnostic F1 score")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def _plot_box_running_best(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """Box-and-whisker plot showing running best score at each round."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = _compute_baseline_metrics()

    for scorer, sdf in round_df.groupby("scorer"):
        # Compute running best for each (module, latent_idx)
        best_rows = []
        rounds = sorted(sdf["round"].unique())
        for (module, latent_idx), g in sdf.groupby(["module", "latent_idx"]):
            for r in rounds:
                best_score = g[g["round"] <= r]["f1_score"].max()
                best_rows.append(
                    {
                        "module": module,
                        "latent_idx": latent_idx,
                        "round": r,
                        "best_f1": best_score,
                    }
                )

        best_df = pd.DataFrame(best_rows)
        output_path = out_dir / f"{scorer}_bnw_runningbest.{image_format}"

        fig, ax = plt.subplots(figsize=(12, 6))
        data_by_round = [
            best_df[best_df["round"] == r]["best_f1"].dropna() for r in rounds
        ]

        bp = ax.boxplot(data_by_round, labels=rounds, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightgreen")

        # Add baseline reference lines
        ax.axhline(y=baselines["always_true_f1"], color="gray", linestyle="--", 
                   linewidth=1, alpha=0.5, label="Always guess positive")
        ax.axhline(y=baselines["random_f1"], color="dimgray", linestyle=":", 
                   linewidth=1, alpha=0.5, label="Random (by frequency)")

        # Titles and labels
        fig.suptitle("Running best frequency-agnostic F1 score by round", fontsize=14, fontweight="bold")
        ax.set_title(run_label, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Best frequency-agnostic F1 score (so far)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def _plot_kde_best_scores(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """KDE of best F1 scores with first round and theoretical max."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = _compute_baseline_metrics()

    for scorer, sdf in round_df.groupby("scorer"):
        # Get best score for each latent
        best_scores = sdf.groupby(["module", "latent_idx"])["f1_score"].max().values

        # Get first round scores (minimum round number)
        min_round = sdf["round"].min()
        first_round = sdf[sdf["round"] == min_round]["f1_score"].dropna().values

        if len(best_scores) < 2 or len(first_round) < 2:
            print(
                f"[plot_kde_best_scores] Insufficient data for {scorer}, skipping KDE"
            )
            continue

        # Compute KDEs
        kde_best = gaussian_kde(best_scores, bw_method=0.3)
        kde_first = gaussian_kde(first_round, bw_method=0.3)

        # Compute plotting range
        x_range = np.linspace(0, 1, 500)
        first_pdf = kde_first(x_range)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))

        best_pdf = kde_best(x_range)
        best_mean = best_scores.mean()
        first_mean = first_round.mean()
        ax.plot(x_range, best_pdf, linewidth=2, label=f"Best (mean={best_mean:.3f})")
        ax.plot(
            x_range,
            first_pdf,
            linewidth=2,
            label=f"First round (mean={first_mean:.3f})",
        )
        # Commented out theoretical max curve per request
        # ax.plot(x_range, theoretical_max_pdf, linewidth=2, linestyle="--", label=f"Theoretical max of {k} IID")

        # Add baseline reference lines
        ax.axvline(x=baselines["always_true_f1"], color="gray", linestyle="--", 
                   linewidth=1, alpha=0.5, label="Always guess positive")
        ax.axvline(x=baselines["random_f1"], color="dimgray", linestyle=":", 
                   linewidth=1, alpha=0.5, label="Random (by frequency)")

        # Titles and labels
        fig.suptitle("Frequency-agnostic F1 score densities (best vs first)", fontsize=14, fontweight="bold")
        ax.set_title(run_label, fontsize=12)
        ax.set_xlabel("Frequency-agnostic F1 score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = out_dir / f"{scorer}_f1KDE_bestscores.{image_format}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def _plot_kde_all_scores(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """KDE of all F1 scores by round with first and theoretical max."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = _compute_baseline_metrics()

    for scorer, sdf in round_df.groupby("scorer"):
        min_round = sdf["round"].min()
        first_round = sdf[sdf["round"] == min_round]["f1_score"].dropna().values

        if len(first_round) < 2:
            print(
                f"[plot_kde_all_scores] Insufficient first-round data for "
                f"{scorer}, skipping"
            )
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        x_range = np.linspace(0, 1, 500)

        # Plot each round
        rounds = sorted(sdf["round"].unique())
        for r in rounds:
            round_scores = sdf[sdf["round"] == r]["f1_score"].dropna().values
            if len(round_scores) >= 2:
                kde_round = gaussian_kde(round_scores, bw_method=0.3)
                round_pdf = kde_round(x_range)
                round_mean = round_scores.mean()
                ax.plot(
                    x_range,
                    round_pdf,
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Round {r} (mean={round_mean:.3f})",
                )

        # First round (emphasized)
        kde_first = gaussian_kde(first_round, bw_method=0.3)
        first_pdf = kde_first(x_range)
        first_mean = first_round.mean()
        """ax.plot(
            x_range,
            first_pdf,
            linewidth=3,
            color="red",
            label=f"First round (mean={first_mean:.3f})",
        )"""
        # Commented out theoretical max curve per request
        # k = len(rounds)
        # theoretical_max_pdf = k * first_pdf * (first_cdf ** (k - 1))
        # ax.plot(x_range, theoretical_max_pdf, linewidth=2.5, linestyle="--", color="black", label=f"Theoretical max of {k} IID")

        # Add baseline reference lines
        ax.axvline(x=baselines["always_true_f1"], color="gray", linestyle="--", 
                   linewidth=1, alpha=0.5, label="Always guess positive")
        ax.axvline(x=baselines["random_f1"], color="dimgray", linestyle=":", 
                   linewidth=1, alpha=0.5, label="Random (by frequency)")

        # Titles and labels
        fig.suptitle("Frequency-agnostic F1 score densities by round", fontsize=14, fontweight="bold")
        ax.set_title(run_label, fontsize=12)
        ax.set_xlabel("Frequency-agnostic F1 score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = out_dir / f"{scorer}_f1KDE_allscores.{image_format}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


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
        append_round_to_prompt: bool = True,
        show_score_to_explainer: bool = False,
        allow_tp_examples: bool = True,
    ) -> ExperimentDefinition:
        cfg = replace(
            self.base_config,
            explainer="iterative",
            iterative_num_rounds=rounds,
            explainer_temperature=0.7,
            iterative_carryforward_strategy=carry,
            iterative_history_only=history_only,
            iterative_always_new_train_examples=always_new_train,
            iterative_append_round_to_prompt=append_round_to_prompt,
            iterative_show_score_to_explainer=show_score_to_explainer,
            iterative_allow_tp_examples=allow_tp_examples,
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
                append_round_to_prompt=True,
                show_score_to_explainer=False,
                allow_tp_examples=True,
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
                append_round_to_prompt=True,
                show_score_to_explainer=False,
                allow_tp_examples=True,
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
                append_round_to_prompt=True,
                show_score_to_explainer=False,
                allow_tp_examples=True,
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
                append_round_to_prompt=True,
                show_score_to_explainer=False,
                allow_tp_examples=True,
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
                append_round_to_prompt=True,
                show_score_to_explainer=False,
                allow_tp_examples=True,
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
                append_round_to_prompt=False,  # History-only experiments don't need round tags
                show_score_to_explainer=True,  # But they do need scores to learn from
                allow_tp_examples=True,
            )
        )
        
        # Variant with 40 train examples per round
        # NOTE: Iterative splits record.train (160) into holdout (16), test (108), train (36)
        # This variant shows 40 examples per round (requires larger n_examples_train)
        cfg_train40 = self._make_iterative_config(
            name="iterative_train40",
            rounds=5,
            carry="best",
            include_tp_tn=True,
            always_new_train=False,
            history_only=False,
            append_round_to_prompt=True,
            show_score_to_explainer=False,
            allow_tp_examples=True,
        )
        cfg_train40 = ExperimentDefinition(
            cfg_train40.name,
            replace(
                cfg_train40.config, 
                iterative_num_train_examples_per_round=40,
                sampler_cfg=replace(cfg_train40.config.sampler_cfg, n_examples_train=160)
            )
        )
        experiments.append(cfg_train40)

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
            name="bestofk_baseline",
        )
        yield ExperimentDefinition(cfg.name, cfg)

        # Add oneshot variant (single prompt generates all K explanations)
        cfg_oneshot = replace(
            cfg,
            bestofk_is_multishot=False,
            name="bestofk_oneshot",
        )
        yield ExperimentDefinition(cfg_oneshot.name, cfg_oneshot)

        # Variant with 40 train examples shown to model
        # NOTE: BestOfK splits record.train (80) into train (20) and test (60)
        # This variant shows 40 examples (requires larger n_examples_train)
        cfg_train40 = replace(
            cfg,
            bestofk_num_train_examples=40,
            sampler_cfg=replace(cfg.sampler_cfg, n_examples_train=160),  # Need 160 to get train pool of ~40
            name="bestofk_train40",
        )
        yield ExperimentDefinition(cfg_train40.name, cfg_train40)

    def build_random_baseline(self, source_run: str = "bestofk_baseline") -> Iterable[ExperimentDefinition]:
        cfg = replace(
            self.base_config,
            explainer="bestofk",
            use_random_baseline=True,
            random_baseline_source_run=source_run,
            name=f"bestofk_random-baseline_from-{source_run}",
        )
        yield ExperimentDefinition(cfg.name, cfg)

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

    def copy_pngs_to_viz(self) -> None:
        """Copy all PNG files from results/{experiment_names}/visualize/ to viz_pngs/{experiment_names}/"""
        viz_root = self.results_root / "viz_pngs"
        viz_root.mkdir(parents=True, exist_ok=True)
        
        # Find all experiment directories in results
        for exp_dir in self.results_root.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "viz_pngs" or exp_dir.name == "visualize-cross-experiment":
                continue
                
            visualize_dir = exp_dir / "visualize"
            if not visualize_dir.exists():
                print(f"[ExperimentBattery] No visualize directory found for {exp_dir.name}")
                continue
                
            # Create corresponding directory in viz_pngs
            viz_exp_dir = viz_root / exp_dir.name
            viz_exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all PNG files
            png_files = list(visualize_dir.glob("*.png"))
            for png_file in png_files:
                dest_path = viz_exp_dir / png_file.name
                shutil.copy2(png_file, dest_path)
                print(f"[ExperimentBattery] Copied {png_file} to {dest_path}")
        
        # Also copy cross-experiment PNGs
        cross_exp_dir = self.results_root / "visualize-cross-experiment"
        if cross_exp_dir.exists():
            viz_cross_dir = viz_root / "cross-experiments"
            viz_cross_dir.mkdir(parents=True, exist_ok=True)
            
            png_files = list(cross_exp_dir.glob("*.png"))
            for png_file in png_files:
                dest_path = viz_cross_dir / png_file.name
                shutil.copy2(png_file, dest_path)
                print(f"[ExperimentBattery] Copied cross-experiment {png_file} to {dest_path}")
        
        print(f"[ExperimentBattery] PNG copying completed. All PNGs copied to {viz_root}")

    def _discover_experiments_from_results(self) -> list[ExperimentDefinition]:
        """Discover all experiments from the results folder by reading run_config.json files."""
        experiments = []
        
        for exp_dir in self.results_root.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "viz_pngs" or exp_dir.name == "visualize-cross-experiment":
                continue
                
            run_config = self._load_run_config(exp_dir.name)
            if run_config is None:
                continue
                
            # Create a basic RunConfig from the loaded data
            # We'll use the base_config as a template and update with the loaded data
            config = replace(
                self.base_config,
                name=exp_dir.name,
                explainer=run_config.get("explainer", "bestofk"),
                # Add other relevant fields as needed
            )
            
            experiments.append(ExperimentDefinition(exp_dir.name, config))
            
        print(f"[ExperimentBattery] Discovered {len(experiments)} experiments from results folder")
        return experiments

    def _load_run_config(self, run_name: str) -> dict | None:
        cfg_path = self._run_dir(run_name) / "run_config.json"
        if not cfg_path.exists():
            print(f"[ExperimentBattery] Missing run_config.json for {run_name}")
            return None
        return orjson.loads(cfg_path.read_bytes())

    def plot_runs(self, experiments: Sequence[ExperimentDefinition]) -> None:
        """Generate standard plots for final results of experiments."""
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

    def plot_multi_round_runs(
        self, experiments: Sequence[ExperimentDefinition]
    ) -> None:
        """Generate multi-round analysis plots for experiments.

        This loads per-round scores from scores/*/multi_scores/ directories
        and generates plots showing:
        - F1 score distribution at each round (box plots)
        - Running best F1 scores over rounds (box plots)
        - KDE plots of best scores vs first round and theoretical max
        - KDE plots of all round scores
        """
        for exp in experiments:
            run_dir = self._run_dir(exp.name)
            cfg_dict = self._load_run_config(exp.name)
            if cfg_dict is None:
                continue

            scores_path = run_dir / "scores"
            visualize_path = run_dir / "visualize"

            if not scores_path.exists():
                print(
                    f"[ExperimentBattery] Scores missing for {exp.name}, "
                    f"skipping multi-round plots"
                )
                continue

            print(f"[ExperimentBattery] Loading multi-round data for {exp.name}...")
            
            # Determine max_rounds cap based on explainer type
            max_rounds = None
            explainer_type = cfg_dict.get("explainer", "default")
            if explainer_type == "bestofk":
                # For BestOfK, cap at K candidates
                max_rounds = cfg_dict.get("bestofk_num_explanations", None)
                if max_rounds:
                    print(f"[ExperimentBattery] Capping BestOfK at {max_rounds} candidates")
            elif explainer_type == "iterative":
                # For Iterative, cap at num_rounds
                max_rounds = cfg_dict.get("iterative_num_rounds", None)
                if max_rounds:
                    print(f"[ExperimentBattery] Capping Iterative at {max_rounds} rounds")
            
            round_df = _load_explanation_scores_per_round(scores_path, max_rounds=max_rounds)

            if round_df.empty:
                print(
                    f"[ExperimentBattery] No multi-round scores found for {exp.name}, "
                    f"skipping multi-round plots"
                )
                continue

            print(
                f"[ExperimentBattery] Loaded {len(round_df)} score records "
                f"for {exp.name}"
            )
            rounds_list = sorted(round_df["round"].unique())
            print(f"[ExperimentBattery] Rounds found: {rounds_list}")
            scorers_list = sorted(round_df["scorer"].unique())
            print(f"[ExperimentBattery] Scorers found: {scorers_list}")

            visualize_path.mkdir(parents=True, exist_ok=True)

            for image_format in ["png", "pdf"]:
                print(
                    f"[ExperimentBattery] Generating {image_format} plots "
                    f"for {exp.name}..."
                )
                _plot_kde_best_scores(round_df, visualize_path, exp.name, image_format)
                _plot_kde_all_scores(round_df, visualize_path, exp.name, image_format)
                _plot_box_running_best(round_df, visualize_path, exp.name, image_format)
                _plot_box_per_round(round_df, visualize_path, exp.name, image_format)

            print(f"[ExperimentBattery] Multi-round plots completed for {exp.name}")

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
        bestofk_experiments: Sequence[ExperimentDefinition] = None,
        iterative_experiments: Sequence[ExperimentDefinition] = None,
        scorers: Sequence[str] = ("fuzz",),
        output_prefix: str = "bestofk_vs_iterative",
    ) -> None:
        import matplotlib.pyplot as plt

        compare_dir = self.results_root / "visualize-cross-experiment"
        compare_dir.mkdir(parents=True, exist_ok=True)

        # If no experiments provided, discover all experiments from results folder
        if bestofk_experiments is None or iterative_experiments is None:
            all_experiments = self._discover_experiments_from_results()
            if bestofk_experiments is None:
                bestofk_experiments = [exp for exp in all_experiments if exp.config.explainer == "bestofk"]
            if iterative_experiments is None:
                iterative_experiments = [exp for exp in all_experiments if exp.config.explainer == "iterative"]

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

                    # Add baseline reference lines (only for F1 scores, not weighted)
                    if metric == "f1_score":
                        baselines = _compute_baseline_metrics()
                        ax.axhline(y=baselines["always_true_f1"], color="gray", linestyle="--", 
                                   linewidth=1, alpha=0.5, label="Always guess positive")
                        ax.axhline(y=baselines["random_f1"], color="dimgray", linestyle=":", 
                                   linewidth=1, alpha=0.5, label="Random (by frequency)")

                    ax.set_xlabel("Experiment", fontsize=12)
                    # Friendly labels for F1 variants
                    y_label = (
                        "Frequency-agnostic F1 score" if metric == "f1_score" else "Frequency-weighted F1 score"
                    )
                    title_label = (
                        "Frequency-agnostic F1 score comparison" if metric == "f1_score" else "Frequency-weighted F1 score comparison"
                    )
                    ax.set_ylabel(y_label, fontsize=12)
                    ax.set_title(f"{title_label} ({scorer})", fontsize=14, fontweight="bold")
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
                        f"[ExperimentBattery] Saved cross-experiment plot: "
                        f"{output_path}"
                    )


if __name__ == "__main__":
    base_cfg = default_run_config()

    base_cfg = replace(
        base_cfg,
        max_latents=200,
        # name will be set per experiment
    )

    # Use absolute path to ensure consistent results location regardless of cwd
    # This resolves to the repo root's results/ directory
    results_dir = (Path(__file__).parent.parent / "results").resolve()
    runner = ExperimentRunner(base_results_dir=results_dir)
    battery = ExperimentBattery(runner, base_cfg)

    # --- ORIGINAL EXPERIMENTS (with improved parameters based on debugging) ---
    bestofk_experiments = list(battery.build_bestofk_grid())
    iterative_experiments = list(battery.build_iterative_grid())

    # Random baseline (runs after its source run exists)
    random_baseline_experiments = list(battery.build_random_baseline("bestofk_baseline"))

    # Easy filter list: include only these experiment names
    include_names: list[str] = [
        "bestofk_baseline",
        "bestofk_oneshot",
        "bestofk_train40",
        "bestofk_random-baseline_from-bestofk_baseline",
        "iterative_baseline",
        "iterative_rounds10",
        "iterative_carry-last",
        "iterative_always-new-train",
        "iterative_no-tp-tn",
        "iterative_history-only",
        "iterative_train40",
    ]
    #include_names = []
    
    PLOT_ONLY = True

    def _maybe_filter(exps: list[ExperimentDefinition]) -> list[ExperimentDefinition]:
        return [x for x in exps if x.name in include_names]

    # Print experiment names for easy commenting
    print("Experiments available:")
    for x in bestofk_experiments + iterative_experiments + random_baseline_experiments:
        print(f" - {x.name}")

    bestofk_experiments = _maybe_filter(bestofk_experiments)
    iterative_experiments = _maybe_filter(iterative_experiments)
    random_baseline_experiments = _maybe_filter(random_baseline_experiments)

    # --- Commands (uncommented to run) ---
    # Run Best-of-K experiments
    print("Running Best-of-K experiments...")
    if not PLOT_ONLY:
        battery.run(bestofk_experiments)

    # Plot Best-of-K runs (final results)
    print("Plotting Best-of-K runs (final results)...")
    battery.plot_runs(bestofk_experiments)

    # Plot Best-of-K multi-round analysis
    print("Plotting Best-of-K multi-round analysis...")
    battery.plot_multi_round_runs(bestofk_experiments)

    # Run Random baseline experiments (requires source runs to exist)
    if random_baseline_experiments:
        print("Running Random Baseline experiments...")
        if not PLOT_ONLY:
            battery.run(random_baseline_experiments)

    # Run Iterative experiments
    print("Running Iterative experiments...")
    if not PLOT_ONLY:
        battery.run(iterative_experiments)

    # Plot Iterative runs (final results)
    print("Plotting Iterative runs (final results)...")
    battery.plot_runs(iterative_experiments)
    if random_baseline_experiments:
        battery.plot_runs(random_baseline_experiments)

    # Plot Iterative multi-round analysis
    print("Plotting Iterative multi-round analysis...")
    battery.plot_multi_round_runs(iterative_experiments)

    # Cross-experiment comparison plots (Best-of-K vs Iterative)
    print("Generating cross-experiment comparison plots...")
    battery.plot_cross_experiments()  # Will discover all experiments automatically

    # Copy all PNGs to viz_pngs folder
    print("Copying PNGs to viz_pngs folder...")
    battery.copy_pngs_to_viz()

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
