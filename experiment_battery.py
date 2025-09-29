"""Run a battery of experiments for Best-of-K vs Iterative explainers."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import orjson
import pandas as pd

from delphi.delphi.config import RunConfig
from delphi.delphi.log.result_analysis import (
    add_latent_f1,
    get_agg_metrics,
    import_plotly,
    load_data,
    log_results,
)
from delphi.run_experiment import ExperimentRunner, default_run_config


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

    def build_iterative_grid(self) -> Iterable[ExperimentDefinition]:
        iterative_rounds = [5, 10]
        carry_strategies = ["last", "best"]
        no_iter_examples = [False, True]
        include_tp_tn = [True, False]
        always_new_train = [False, True]

        for rounds in iterative_rounds:
            for carry in carry_strategies:
                for history_only in no_iter_examples:
                    tp_tn_opts = include_tp_tn if not history_only else [False]
                    new_train_opts = always_new_train if not history_only else [False]
                    for allow_tp_tn in tp_tn_opts:
                        for new_train in new_train_opts:
                            cfg = replace(
                                self.base_config,
                                explainer="iterative",
                                iterative_num_rounds=rounds,
                                explainer_temperature=0.7,
                                iterative_carryforward_strategy=carry,
                                iterative_history_only=history_only,
                                iterative_max_num_true_positives=(
                                    self.base_config.iterative_max_num_true_positives
                                    if allow_tp_tn
                                    else 0
                                ),
                                iterative_max_num_true_negatives=(
                                    self.base_config.iterative_max_num_true_negatives
                                    if allow_tp_tn
                                    else 0
                                ),
                                iterative_always_new_train_examples=new_train,
                                name=(
                                    f"iterative_rounds{rounds}_carry-{carry}_"
                                    f"no_iterative_examples-{history_only}_"
                                    f"tp_tn-{allow_tp_tn}_new_train-{new_train}"
                                ),
                            )
                            yield ExperimentDefinition(cfg.name, cfg)

    def build_bestofk_grid(self) -> Iterable[ExperimentDefinition]:
        cfg = replace(
            self.base_config,
            explainer="bestofk",
            bestofk_num_explanations=10,
            explainer_temperature=0.7,
            judge_scorer_index=self.base_config.scorers.index("fuzz")
            if "fuzz" in self.base_config.scorers
            else 0,
            name="bestofk_k10_temp0.7_judge_fuzz",
        )
        yield ExperimentDefinition(cfg.name, cfg)

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
        compare_dir = self.results_root / "comparisons"
        compare_dir.mkdir(parents=True, exist_ok=True)
        px = import_plotly()

        for scorer in scorers:
            bok_df = self._collect_metrics(bestofk_experiments, scorer)
            it_df = self._collect_metrics(iterative_experiments, scorer)
            combined = pd.concat([bok_df, it_df], ignore_index=True)
            if combined.empty:
                print(f"[ExperimentBattery] No metrics available for scorer '{scorer}'")
                continue
            combined["order"] = range(len(combined))
            fig = px.scatter(
                combined,
                x="order",
                y="f1_score",
                color="explainer",
                symbol="explainer",
                hover_data={
                    "run": True,
                    "score_type": True,
                    "weighted_f1": True,
                    "order": False,
                },
            )
            fig.update_traces(marker=dict(size=10))
            fig.update_layout(
                title=f"F1 comparison ({scorer})",
                xaxis_title="Experiment",
                yaxis_title="F1 Score",
                xaxis=dict(
                    tickmode="array",
                    tickvals=combined["order"].tolist(),
                    ticktext=combined["run"].tolist(),
                    tickangle=60,
                ),
                legend_title="Explainer",
            )
            for fmt in ("png", "pdf"):
                fig.write_image(compare_dir / f"{output_prefix}_{scorer}.{fmt}")


if __name__ == "__main__":
    base_cfg = default_run_config()

    base_cfg = replace(
        base_cfg,
        max_latents=20,
        name="iterative_baseline",
    )

    runner = ExperimentRunner()
    battery = ExperimentBattery(runner, base_cfg)

    bestofk_experiments = list(battery.build_bestofk_grid())
    iterative_experiments = list(battery.build_iterative_grid())

    # --- Commands (comment out to skip) ---
    # Run Best-of-K baseline experiments
    battery.run(bestofk_experiments)

    # Plot Best-of-K runs
    battery.plot_runs(bestofk_experiments)

    # Run Iterative experiments
    battery.run(iterative_experiments)

    # Plot Iterative runs
    battery.plot_runs(iterative_experiments)

    # Cross-experiment comparison plots (Best-of-K vs Iterative)
    battery.plot_cross_experiments(bestofk_experiments, iterative_experiments)
