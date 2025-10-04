import re
from pathlib import Path
from typing import Literal, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


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

        output_path = out_dir / f"{run_label}_{module}_firing_rates.{image_format}"

        # Use matplotlib for PNG/PDF
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(module_df["firing_rate"], module_df["f1_score"], alpha=0.6, s=20)
        ax.set_xscale("log")
        ax.set_xlim(10**-5.4, module_df["firing_rate"].max() * 1.1)
        ax.set_xlabel("Firing rate")
        ax.set_ylabel("F1 score")
        ax.set_title(f"{run_label} - {module}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def compute_auc(df: pd.DataFrame) -> float | None:
    if not df.probability.nunique():
        return None

    valid_df = df[df.probability.notna()]

    return roc_auc_score(valid_df.activating, valid_df.probability)  # type: ignore


def plot_accuracy_hist(df: pd.DataFrame, out_dir: Path, image_format: str = "pdf"):
    """Histogram of accuracy for the BEST explanation per latent."""
    out_dir.mkdir(exist_ok=True, parents=True)
    for label in df["score_type"].unique():
        output_path = (
            out_dir / f"best_explanation_{label}_accuracy_histogram.{image_format}"
        )
        subset = df[df["score_type"] == label]

        # Use matplotlib for PNG/PDF
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(subset["accuracy"], bins=100, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Accuracy distribution of BEST {label} explanations across latents"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_roc_curve(df: pd.DataFrame, out_dir: Path, image_format: str = "pdf"):
    if not df.probability.nunique():
        return

    # filter out NANs
    valid_df = df[df.probability.notna()]

    fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
    auc = roc_auc_score(valid_df.activating, valid_df.probability)
    out_dir.mkdir(exist_ok=True, parents=True)
    output_path = out_dir / f"roc_curve.{image_format}"

    # Use matplotlib for PNG/PDF
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_weighted_f1_bar(
    processed_df: pd.DataFrame, out_dir: Path, image_format: str = "pdf"
) -> None:
    if processed_df.empty or "weighted_f1" not in processed_df.columns:
        return
    safe_df = processed_df.dropna(subset=["weighted_f1"])
    if safe_df.empty:
        return
    out_dir.mkdir(exist_ok=True, parents=True)
    output_path = out_dir / f"weighted_f1_by_scorer.{image_format}"

    # Use matplotlib for PNG/PDF
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(safe_df)), safe_df["weighted_f1"], alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(safe_df)))
    ax.set_xticklabels(safe_df["score_type"], rotation=45, ha="right")
    ax.set_xlabel("Scorer")
    ax.set_ylabel("Weighted F1")
    ax.set_title("Weighted F1 by scorer")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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
    if len(latent_dfs) > 1:
        return pd.concat(latent_dfs, ignore_index=True), counts
    else:
        return latent_dfs[0], counts


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
    """Add latent-level F1 scores to the dataframe."""
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
        multi_files = list(multi_dir.glob("*.txt"))
        if not multi_dir.exists() or len(multi_files) == 0:
            msg = (
                f"[load_explanation_scores_per_round] No multi_scores "
                f"found for {scorer_dir.name}, skipping"
            )
            print(msg)
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


def plot_box_per_round(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """Box-and-whisker plot showing score distribution at each round."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for scorer, sdf in round_df.groupby("scorer"):
        output_path = out_dir / f"{scorer}_bnw_perround.{image_format}"

        fig, ax = plt.subplots(figsize=(12, 6))
        rounds = sorted(sdf["round"].unique())
        data_by_round = [sdf[sdf["round"] == r]["f1_score"].dropna() for r in rounds]

        bp = ax.boxplot(data_by_round, labels=rounds, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax.set_xlabel("Round")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"F1 Score Distribution by Round - {scorer}")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_box_running_best(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """Box-and-whisker plot showing running best score at each round."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

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

        ax.set_xlabel("Round")
        ax.set_ylabel("Best F1 Score (so far)")
        ax.set_title(f"Running Best F1 Score by Round - {scorer}")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_kde_best_scores(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """KDE of best F1 scores with first round and theoretical max."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for scorer, sdf in round_df.groupby("scorer"):
        # Get best score for each latent
        best_scores = sdf.groupby(["module", "latent_idx"])["f1_score"].max().values

        # Get first round scores (minimum round number)
        min_round = sdf["round"].min()
        first_round = sdf[sdf["round"] == min_round]["f1_score"].dropna().values

        if len(best_scores) < 2 or len(first_round) < 2:
            continue

        # Compute KDEs
        kde_best = gaussian_kde(best_scores, bw_method=0.3)
        kde_first = gaussian_kde(first_round, bw_method=0.3)

        # Theoretical max distribution from first round
        x_range = np.linspace(0, 1, 500)
        first_pdf = kde_first(x_range)
        first_cdf = np.array([kde_first.integrate_box_1d(0, x) for x in x_range])

        k = sdf["round"].max() + 1  # Number of candidates
        theoretical_max_pdf = k * first_pdf * (first_cdf ** (k - 1))

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
        ax.plot(
            x_range,
            theoretical_max_pdf,
            linewidth=2,
            linestyle="--",
            label=f"Theoretical max of {k} IID",
        )

        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Density")
        ax.set_title(f"F1 Score Distribution - Best Explanations - {scorer}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = out_dir / f"{scorer}_f1KDE_bestscores.{image_format}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_kde_all_scores(
    round_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    image_format: str = "pdf",
) -> None:
    """KDE of all F1 scores by round with first and theoretical max."""
    if round_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for scorer, sdf in round_df.groupby("scorer"):
        min_round = sdf["round"].min()
        first_round = sdf[sdf["round"] == min_round]["f1_score"].dropna().values

        if len(first_round) < 2:
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
        first_cdf = np.array([kde_first.integrate_box_1d(0, x) for x in x_range])
        first_mean = first_round.mean()
        ax.plot(
            x_range,
            first_pdf,
            linewidth=3,
            color="red",
            label=f"First round (mean={first_mean:.3f})",
        )

        # Theoretical max
        k = len(rounds)
        theoretical_max_pdf = k * first_pdf * (first_cdf ** (k - 1))
        ax.plot(
            x_range,
            theoretical_max_pdf,
            linewidth=2.5,
            linestyle="--",
            color="black",
            label=f"Theoretical max of {k} IID",
        )

        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Density")
        ax.set_title(f"F1 Score Distribution - All Rounds - {scorer}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = out_dir / f"{scorer}_f1KDE_allscores.{image_format}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def analyze_explainer_rounds(
    scores_path: Path,
    out_dir: Path,
    run_label: Optional[str] = None,
    image_format: str = "pdf",
) -> None:
    """Load per-round scores and generate all round-based plots."""
    if run_label is None:
        run_label = scores_path.name

    round_df = load_explanation_scores_per_round(scores_path)
    if round_df.empty:
        print("No per-round scores found. Skipping round-based plots.")
        return

    # Generate all 4 plot types
    plot_kde_best_scores(round_df, out_dir, run_label, image_format)
    plot_kde_all_scores(round_df, out_dir, run_label, image_format)
    plot_box_running_best(round_df, out_dir, run_label, image_format)
    plot_box_per_round(round_df, out_dir, run_label, image_format)


def log_results(
    scores_path: Path,
    viz_path: Path,
    modules: list[str],
    scorer_names: list[str],
    image_formats: list[Literal["png", "pdf"]] = ["pdf"],
):
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
        # Skip weighted_f1_bar - user doesn't want this

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
            try:
                analyze_explainer_rounds(
                    scores_path,
                    viz_path,
                    run_label=scores_path.name,
                    image_format=image_format,
                )
            except Exception as e:
                print(f"Warning: Could not generate round-based plots: {e}")
                print("This is expected if multi_scores data is missing.")
