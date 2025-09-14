import re
from pathlib import Path
from typing import Literal, Optional

import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def plot_firing_vs_f1(
    latent_df: pd.DataFrame,
    num_tokens: int,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for module, module_df in latent_df.groupby("module"):
        module_df = module_df.copy()
        module_df["firing_rate"] = module_df["firing_count"] / num_tokens
        fig = px.scatter(module_df, x="firing_rate", y="f1_score", log_x=True)
        fig.update_layout(
            xaxis_title="Firing rate", yaxis_title="F1 score", xaxis_range=[-5.4, 0]
        )
        fig.write_image(out_dir / f"{run_label}_{module}_firing_rates.{image_format}")


def import_plotly():
    """Import plotly with mitigiation for MathJax bug."""
    try:
        import plotly.express as px  # type: ignore
        import plotly.io as pio  # type: ignore
    except ImportError:
        raise ImportError(
            "Plotly is not installed.\n"
            "Please install it using `pip install plotly`, "
            "or install the `[visualize]` extra."
        )
    pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
    return px


def compute_auc(df: pd.DataFrame) -> float | None:
    if not df.probability.nunique():
        return None

    valid_df = df[df.probability.notna()]

    return roc_auc_score(valid_df.activating, valid_df.probability)  # type: ignore


def plot_accuracy_hist(df: pd.DataFrame, out_dir: Path, image_format: str = "pdf"):
    out_dir.mkdir(exist_ok=True, parents=True)
    for label in df["score_type"].unique():
        fig = px.histogram(
            df[df["score_type"] == label],
            x="accuracy",
            nbins=100,
            title=f"Accuracy distribution: {label}",
        )
        fig.write_image(out_dir / f"{label}_accuracy.{image_format}")


def plot_roc_curve(df: pd.DataFrame, out_dir: Path, image_format: str = "pdf"):
    if not df.probability.nunique():
        return

    # filter out NANs
    valid_df = df[df.probability.notna()]

    fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
    auc = roc_auc_score(valid_df.activating, valid_df.probability)
    fig = go.Figure(
        data=[
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")),
        ]
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="FPR",
        yaxis_title="TPR",
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.write_image(out_dir / f"roc_curve.{image_format}")


def compute_confusion(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    df_valid = df[df["prediction"].notna()]
    act = df_valid["activating"].astype(bool)

    total = len(df_valid)
    pos = act.sum()
    neg = total - pos

    tp = ((df_valid.prediction >= threshold) & act).sum()
    tn = ((df_valid.prediction < threshold) & ~act).sum()
    fp = ((df_valid.prediction >= threshold) & ~act).sum()
    fn = ((df_valid.prediction < threshold) & act).sum()

    assert fp <= neg and tn <= neg and tp <= pos and fn <= pos

    return dict(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_examples=total,
        total_positives=pos,
        total_negatives=neg,
        failed_count=len(df_valid) - total,
    )


def compute_classification_metrics(conf: dict) -> dict:
    tp = conf["true_positives"]
    tn = conf["true_negatives"]
    fp = conf["false_positives"]
    fn = conf["false_negatives"]
    total = conf["total_examples"]
    pos = conf["total_positives"]
    neg = conf["total_negatives"]

    assert pos + neg == total, "pos + neg must equal total"

    # accuracy = (tp + tn) / total if total > 0 else 0
    balanced_accuracy = (
        (tp / pos if pos > 0 else 0) + (tn / neg if neg > 0 else 0)
    ) / 2

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / pos if pos > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return dict(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=balanced_accuracy,
        true_positive_rate=tp / pos if pos > 0 else 0,
        true_negative_rate=tn / neg if neg > 0 else 0,
        false_positive_rate=fp / neg if neg > 0 else 0,
        false_negative_rate=fn / pos if pos > 0 else 0,
        total_examples=total,
        total_positives=pos,
        total_negatives=neg,
        positive_class_ratio=pos / total if total > 0 else 0,
        negative_class_ratio=neg / total if total > 0 else 0,
    )


def load_data(scores_path: Path, modules: list[str]):
    """Load all on-disk data into a single DataFrame."""

    def parse_score_file(path: Path) -> pd.DataFrame:
        """
        Load a score file and return a raw DataFrame
        """
        try:
            data = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            print(f"Error decoding JSON from {path}. Skipping file.")
            return pd.DataFrame()

        latent_idx = int(path.stem.split("latent")[-1])

        return pd.DataFrame(
            [
                {
                    "text": "".join(ex["str_tokens"]),
                    "distance": ex["distance"],
                    "activating": ex["activating"],
                    "prediction": ex["prediction"],
                    "probability": ex["probability"],
                    "correct": ex["correct"],
                    "activations": ex["activations"],
                    "latent_idx": latent_idx,
                }
                for ex in data
            ]
        )

    counts_file = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
    counts = torch.load(counts_file, weights_only=True) if counts_file.exists() else {}
    if not all(module in counts for module in modules):
        print("Missing firing counts for some modules, setting counts to None.")
        print(f"Missing modules: {[m for m in modules if m not in counts]}")
        counts = None

    # Collect per-latent data
    latent_dfs = []
    for score_type_dir in scores_path.iterdir():
        if not score_type_dir.is_dir():
            continue
        for module in modules:
            for file in score_type_dir.glob(f"*{module}*"):
                latent_idx = int(file.stem.split("latent")[-1])

                latent_df = parse_score_file(file)
                latent_df["score_type"] = score_type_dir.name
                latent_df["module"] = module
                latent_df["latent_idx"] = latent_idx
                if counts:
                    latent_df["firing_count"] = (
                        counts[module][latent_idx].item()
                        if latent_idx in counts[module]
                        else None
                    )

                latent_dfs.append(latent_df)

    return pd.concat(latent_dfs, ignore_index=True), counts


def frequency_weighted_f1(
    df: pd.DataFrame, counts: dict[str, torch.Tensor]
) -> float | None:
    rows = []
    for (module, latent_idx), grp in df.groupby(["module", "latent_idx"]):
        f1 = compute_classification_metrics(compute_confusion(grp))["f1_score"]
        fire = counts[module][latent_idx].item()
        rows.append(
            {
                "module": module,
                "latent_idx": latent_idx,
                "f1_score": f1,
                "firing_count": fire,
            }
        )

    latent_df = pd.DataFrame(rows)

    per_module_f1 = []
    for module in latent_df["module"].unique():
        module_df = latent_df[latent_df["module"] == module]

        firing_weights = counts[module][module_df["latent_idx"]].float()
        total_weight = firing_weights.sum()
        if total_weight == 0:
            continue

        f1_tensor = torch.as_tensor(module_df["f1_score"].values, dtype=torch.float32)
        module_f1 = (f1_tensor * firing_weights).sum() / firing_weights.sum()
        per_module_f1.append(module_f1)

    overall_frequency_weighted_f1 = torch.stack(per_module_f1).mean()
    return (
        overall_frequency_weighted_f1.item()
        if not overall_frequency_weighted_f1.isnan()
        else None
    )


def get_agg_metrics(
    latent_df: pd.DataFrame, counts: Optional[dict[str, torch.Tensor]]
) -> pd.DataFrame:
    processed_rows = []
    for score_type, group_df in latent_df.groupby("score_type"):
        conf = compute_confusion(group_df)
        class_m = compute_classification_metrics(conf)
        auc = compute_auc(group_df)
        f1_w = frequency_weighted_f1(group_df, counts) if counts else None

        row = {
            "score_type": score_type,
            **conf,
            **class_m,
            "auc": auc,
            "weighted_f1": f1_w,
        }
        processed_rows.append(row)

    return pd.DataFrame(processed_rows)


def add_latent_f1(latent_df: pd.DataFrame) -> pd.DataFrame:
    f1s = (
        latent_df.groupby(["module", "latent_idx"])
        .apply(
            lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
        )
        .reset_index(name="f1_score")  # <- naive (un-weighted) F1
    )
    return latent_df.merge(f1s, on=["module", "latent_idx"])


# ----------------------
# Round-wise explanation analysis (Best-of-K, Iterative)
# ----------------------


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


def load_explanation_scores_per_round(scores_path: Path) -> pd.DataFrame:
    """Load per-explanation (per-round) scores from scores/*/multi_scores.

    Returns a DataFrame with columns:
    - module, latent_idx, round, scorer, f1_score, accuracy, precision, recall
    """
    rows = []
    for scorer_dir in scores_path.iterdir():
        if not scorer_dir.is_dir():
            continue
        multi_dir = scorer_dir / "multi_scores"
        if not multi_dir.exists():
            # Not all explainers persist multi_scores (e.g., iterative in some runs)
            continue
        for file in multi_dir.glob("*.txt"):
            try:
                module, latent_idx, round_idx = _parse_multi_score_filename(file.stem)
            except ValueError:
                continue

            df = load_single_score_file(file)
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


def load_single_score_file(path: Path) -> pd.DataFrame:
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


def plot_round_box_and_bar(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
    metric: str = "f1_score",
) -> None:
    """Create box (whisker) and mean-with-error bar plots for a metric per round.

    Expects round_df with columns: round, scorer, and the metric (e.g., f1_score).
    """
    if round_df.empty:
        print("No per-round scores found (multi_scores missing). Skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for scorer, sdf in round_df.groupby("scorer"):
        # Box (whisker) plot per round
        fig_box = px.box(
            sdf,
            x="round",
            y=metric,
            points=False,
            title=f"{run_label} - {scorer} - {metric} per round",
        )
        fig_box.update_layout(xaxis_title="Round", yaxis_title=metric)
        fig_box.write_image(
            out_dir / f"{run_label}_{scorer}_{metric}_per_round_box.{image_format}"
        )

        # Bar with whiskers (mean ± std) per round
        agg = (
            sdf.groupby("round")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": metric})
        )
        fig_bar = px.bar(
            agg,
            x="round",
            y=metric,
            title=f"{run_label} - {scorer} - mean {metric} per round",
            error_y="std",
        )
        fig_bar.update_layout(xaxis_title="Round", yaxis_title=f"mean {metric}")
        fig_bar.write_image(
            out_dir / f"{run_label}_{scorer}_{metric}_per_round_bar.{image_format}"
        )


def plot_best_so_far_box_and_bar(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
    metric: str = "f1_score",
) -> None:
    """Box and bar plots for the best-so-far metric at each round.

    For each (module, latent_idx), compute best metric up to and including round r,
    then plot distribution across latents for each r.
    """
    if round_df.empty:
        print("No per-round scores found (multi_scores missing). Skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    best_rows = []
    for scorer, sdf in round_df.groupby("scorer"):
        rounds = sorted(sdf["round"].unique())
        for (module, latent_idx), g in sdf.groupby(["module", "latent_idx"]):
            g_sorted = g.sort_values("round")
            best_val = None
            best_map = {}
            for r in rounds:
                sub = g_sorted[g_sorted["round"] <= r]
                if sub.empty:
                    continue
                current_best = sub[metric].max()
                best_map[r] = (
                    current_best if best_val is None else max(best_val, current_best)
                )
                best_val = best_map[r]
            for r, val in best_map.items():
                best_rows.append(
                    {
                        "module": module,
                        "latent_idx": latent_idx,
                        "round": r,
                        "scorer": scorer,
                        metric: val,
                    }
                )

    best_df = pd.DataFrame(best_rows)
    if best_df.empty:
        print("No best-so-far data computed. Skipping plots.")
        return

    for scorer, sdf in best_df.groupby("scorer"):
        fig_box = px.box(
            sdf,
            x="round",
            y=metric,
            points=False,
            title=f"{run_label} - {scorer} - best-so-far {metric} per round",
        )
        fig_box.update_layout(xaxis_title="Round", yaxis_title=metric)
        fig_box.write_image(
            out_dir / f"{run_label}_{scorer}_bestsofar_{metric}_box.{image_format}"
        )

        agg = (
            sdf.groupby("round")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": metric})
        )
        fig_bar = px.bar(
            agg,
            x="round",
            y=metric,
            title=f"{run_label} - {scorer} - best-so-far mean {metric} per round",
            error_y="std",
        )
        fig_bar.update_layout(xaxis_title="Round", yaxis_title=f"mean {metric}")
        fig_bar.write_image(
            out_dir / f"{run_label}_{scorer}_bestsofar_{metric}_bar.{image_format}"
        )


def analyze_explainer_rounds(
    scores_path: Path,
    out_dir: Path,
    run_label: Optional[str] = None,
    image_format: str = "pdf",
    metric: str = "f1_score",
) -> None:
    """Convenience wrapper: load per-round scores and emit both plots.

    This expects per-explanation scores saved under scores/*/multi_scores as JSON
    (the format produced when best-of-k writes multi_scores). For iterative runs,
    ensure per-round scores are persisted similarly to enable these plots.
    """
    if run_label is None:
        run_label = scores_path.name
    round_df = load_explanation_scores_per_round(scores_path)
    plot_round_box_and_bar(round_df, out_dir, run_label, image_format, metric)
    plot_best_so_far_box_and_bar(round_df, out_dir, run_label, image_format, metric)


def log_results(
    scores_path: Path,
    viz_path: Path,
    modules: list[str],
    scorer_names: list[str],
    image_formats: list[Literal["png", "pdf"]] = ["pdf"],
):
    import_plotly()

    latent_df, counts = load_data(scores_path, modules)
    latent_df = latent_df[latent_df["score_type"].isin(scorer_names)]
    latent_df = add_latent_f1(latent_df)

    for image_format in image_formats:
        plot_firing_vs_f1(
            latent_df,
            num_tokens=10_000_000,
            out_dir=viz_path,
            run_label=scores_path.name,
            image_format=image_format,
        )

    if latent_df.empty:
        print("No data found")
        return

    dead = sum((counts[m] == 0).sum().item() for m in modules)
    print(f"Number of dead features: {dead}")
    print(f"Number of interpreted live features: {len(latent_df)}")

    # Load constructor config for run
    with open(scores_path.parent / "run_config.json", "r") as f:
        run_cfg = orjson.loads(f.read())
    constructor_cfg = run_cfg.get("constructor_cfg", {})
    min_examples = constructor_cfg.get("min_examples", None)
    print("min examples", min_examples)

    if min_examples is not None:
        uninterpretable_features = sum(
            [(counts[m] < min_examples).sum() for m in modules]
        )
        print(
            f"Number of features below the interpretation firing"
            f" count threshold: {uninterpretable_features}"
        )
    for image_format in image_formats:
        plot_roc_curve(latent_df, viz_path, image_format=image_format)

    processed_df = get_agg_metrics(latent_df, counts)

    # Plot per-latent accuracy distributions rather than aggregated single-row metrics
    latent_acc_df = (
        latent_df.groupby(["score_type", "module", "latent_idx"])[  # per latent
            "correct"
        ]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    for image_format in image_formats:
        plot_accuracy_hist(latent_acc_df, viz_path, image_format=image_format)

    for score_type in processed_df.score_type.unique():
        score_type_summary = processed_df[processed_df.score_type == score_type].iloc[0]
        print(f"\n--- {score_type.title()} Metrics ---")
        print(f"Class-Balanced Accuracy: {score_type_summary['accuracy']:.3f}")
        print(f"F1 Score: {score_type_summary['f1_score']:.3f}")
        print(f"Frequency-Weighted F1 Score: {score_type_summary['weighted_f1']:.3f}")
        print(
            "Note: the frequency-weighted F1 score is computed over each"
            " hookpoint and averaged"
        )
        print(f"Precision: {score_type_summary['precision']:.3f}")
        print(f"Recall: {score_type_summary['recall']:.3f}")
        # Only print AUC if unbalanced AUC is not -1.
        if score_type_summary["auc"] is not None:
            print(f"AUC: {score_type_summary['auc']:.3f}")
        else:
            print("Logits not available.")

        fractions_failed = [
            score_type_summary["failed_count"]
            / (
                score_type_summary["total_examples"]
                + score_type_summary["failed_count"]
            )
        ]
        print(
            f"""Average fraction of failed examples: \
{sum(fractions_failed) / len(fractions_failed)}"""
        )

        print("\nConfusion Matrix:")
        print(
            f"True Positive Rate:  {score_type_summary['true_positive_rate']:.3f} "
            f"({score_type_summary['true_positives'].sum()})"
        )
        print(
            f"True Negative Rate:  {score_type_summary['true_negative_rate']:.3f} "
            f"({score_type_summary['true_negatives'].sum()})"
        )
        print(
            f"False Positive Rate: {score_type_summary['false_positive_rate']:.3f} "
            f"({score_type_summary['false_positives'].sum()})"
        )
        print(
            f"False Negative Rate: {score_type_summary['false_negative_rate']:.3f} "
            f"({score_type_summary['false_negatives'].sum()})"
        )

        print("\nClass Distribution:")
        print(f"""Positives: {score_type_summary["total_positives"].sum():.0f}""")
        print(f"""Negatives: {score_type_summary["total_negatives"].sum():.0f}""")
        print(f"Total: {score_type_summary['total_examples'].sum():.0f}")

    # Additionally, for best-of-k or iterative explainers, generate per-round plots
    try:
        with open(scores_path.parent / "run_config.json", "r") as f:
            run_cfg = orjson.loads(f.read())
        explainer_name = run_cfg.get("explainer", "")
    except Exception:
        explainer_name = ""

    if explainer_name in ("bestofk", "iterative"):
        for image_format in image_formats:
            analyze_explainer_rounds(
                scores_path,
                viz_path,
                run_label=scores_path.name,
                image_format=image_format,
                metric="f1_score",
            )
