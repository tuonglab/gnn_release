from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit


# ---------------------------------------------------------------------
# Boxplot of individual sample scores
# ---------------------------------------------------------------------
def boxplot_individual_sample(
    scores: np.ndarray | list[float],
    save: bool = False,
    out_path: str | Path | None = None,
) -> None:
    """
    Draw a boxplot of per item scores.

    Args:
        scores: Array or list of numeric scores.
        save: If True, save the figure to out_path.
        out_path: File path where the figure is saved when save is True.

    Returns:
        None
    """
    if save and out_path is None:
        raise ValueError("out_path must be provided when save is True")

    fig = plt.figure(figsize=(24, 10))
    sns.boxplot(data=scores, orientation="vertical")
    plt.title("Boxplots of Individual Sample Scores")
    plt.ylabel("Score")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        plt.savefig(out_path, bbox_inches="tight")  # type: ignore[arg-type]

    plt.close(fig)


# ---------------------------------------------------------------------
# Scatterplot of individual sample scores
# ---------------------------------------------------------------------
def scatterplot_individual_sample(
    scores: np.ndarray | list[float],
    save: bool = False,
    out_path: str | Path | None = None,
) -> None:
    """
    Draw a scatterplot of per item scores against index.

    Args:
        scores: Array or list of numeric scores.
        save: If True, save the figure to out_path.
        out_path: File path where the figure is saved when save is True.

    Returns:
        None
    """
    if save and out_path is None:
        raise ValueError("out_path must be provided when save is True")

    fig = plt.figure(figsize=(24, 10))
    sns.scatterplot(x=np.arange(len(scores)), y=scores, s=80)
    plt.title("Scatterplot of Individual Sample Scores")
    plt.ylabel("Score")
    plt.xlabel("Index")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        plt.savefig(out_path, bbox_inches="tight")  # type: ignore[arg-type]

    plt.close(fig)


# ---------------------------------------------------------------------
# Aggregate inv logit per source and plot two charts
# ---------------------------------------------------------------------
def plot_inv_logit_per_source(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute inverse logit per row, summarize per source, and plot distribution and means.

    Expects columns sequence, scores, source. Produces a boxplot of per row inv logit
    by source and a scatterplot of per source means.

    Args:
        df: Input table with columns sequence, scores, source.
        save: If True, save plots into out_dir.
        out_dir: Directory to write figures when save is True.

    Returns:
        DataFrame with columns source and inv_logit_mean, sorted by mean descending.
    """
    required = {"sequence", "scores", "source"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required}")

    df = df.copy()
    df["inv_logit"] = expit(df["scores"].astype(float))

    summary = (
        df.groupby("source", as_index=False)["inv_logit"]
        .mean()
        .rename(columns={"inv_logit": "inv_logit_mean"})
        .sort_values("inv_logit_mean", ascending=False)
        .reset_index(drop=True)
    )

    if save:
        if out_dir is None:
            raise ValueError("out_dir must be provided when save is True")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # boxplot
    fig1 = plt.figure(figsize=(24, 10))
    sns.boxplot(data=df, x="source", y="inv_logit", orientation="vertical")
    plt.title("Inverse Logit Score Distribution per Source")
    plt.ylabel("Inverse logit score")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir / "inv_logit_boxplot.png", bbox_inches="tight")  # type: ignore[operator]

    plt.close(fig1)

    # scatterplot of means
    fig2 = plt.figure(figsize=(24, 10))
    sns.scatterplot(data=summary, x="source", y="inv_logit_mean", s=200)
    plt.title("Inverse Logit Mean per Source")
    plt.ylabel("Inverse logit mean")
    plt.xlabel("Source")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir / "inv_logit_mean_scatterplot.png", bbox_inches="tight")  # type: ignore[operator]

    plt.close(fig2)

    return summary


# ---------------------------------------------------------------------
# Cancer vs Control
# ---------------------------------------------------------------------
def summarize_and_plot_inv_logit_means(
    cancer_df: pd.DataFrame,
    control_df: pd.DataFrame,
    save: bool = False,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize inverse logit means per source for two groups and plot a boxplot.

    Args:
        cancer_df: Table for the positive group with columns sequence, scores, source.
        control_df: Table for the negative group with columns sequence, scores, source.
        save: If True, save the boxplot in out_dir.
        out_dir: Directory to write the figure when save is True.

    Returns:
        summary_long: Long table with columns source, group, inv_logit_mean.
        summary_wide: Wide table with columns source and one column per group.
    """
    required = {"sequence", "scores", "source"}
    if not required.issubset(cancer_df.columns):
        raise ValueError(f"cancer_df must contain columns: {required}")
    if not required.issubset(control_df.columns):
        raise ValueError(f"control_df must contain columns: {required}")

    c_df = cancer_df.copy()
    n_df = control_df.copy()

    c_df["inv_logit"] = expit(c_df["scores"].astype(float))
    n_df["inv_logit"] = expit(n_df["scores"].astype(float))

    c_summary = (
        c_df.groupby("source", as_index=False)["inv_logit"]
        .mean()
        .rename(columns={"inv_logit": "inv_logit_mean"})
    )
    c_summary["group"] = "Cancer"

    n_summary = (
        n_df.groupby("source", as_index=False)["inv_logit"]
        .mean()
        .rename(columns={"inv_logit": "inv_logit_mean"})
    )
    n_summary["group"] = "Control"

    summary_long = (
        pd.concat([c_summary, n_summary], ignore_index=True)[
            ["source", "group", "inv_logit_mean"]
        ]
        .sort_values(["group", "inv_logit_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    summary_wide = (
        summary_long.pivot_table(
            index="source", columns="group", values="inv_logit_mean", aggfunc="first"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    if save:
        if out_dir is None:
            raise ValueError("out_dir must be provided when save is True")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))
    sns.boxplot(
        data=summary_long, x="group", y="inv_logit_mean", orientation="vertical"
    )
    plt.title("Per Source Inverse Logit Means: Cancer vs Control")
    plt.ylabel("Inverse logit mean per source")
    plt.xlabel("")
    plt.tight_layout()

    if save:
        plt.savefig(
            out_dir / "inv_logit_mean_cancer_vs_control_boxplot.png",  # type: ignore[operator]
            bbox_inches="tight",
        )

    plt.close(fig)

    return summary_long, summary_wide


# ---------------------------------------------------------------------
# ROC plotting
# ---------------------------------------------------------------------
def plot_roc_from_summary(
    summary_long: pd.DataFrame,
    positive_group: str = "Cancer",
    score_col: str = "inv_logit_mean",
    group_col: str = "group",
    save: bool = False,
    out_path: str | Path | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Compute ROC points and AUC from a summary table with group labels and scores.

    Args:
        summary_long: Long table containing at least group_col and score_col.
        positive_group: Label to be treated as the positive class.
        score_col: Column containing numeric scores.
        group_col: Column containing group labels.
        save: If True, save the ROC figure to out_path.
        out_path: File path to save the figure when save is True.

    Returns:
        roc_df: DataFrame with columns fpr, tpr, threshold including endpoints.
        auc_value: Area under the ROC curve in [0, 1].
    """
    if group_col not in summary_long.columns or score_col not in summary_long.columns:
        raise ValueError(f"summary_long must contain '{group_col}' and '{score_col}'")
    if save and out_path is None:
        raise ValueError("out_path must be provided when save is True")

    df = summary_long[[group_col, score_col]].dropna().copy()
    if df.empty:
        raise ValueError("No data after dropping NaNs")

    y_true = (df[group_col].astype(str).values == str(positive_group)).astype(int)
    y_score = df[score_col].astype(float).values

    P = int(y_true.sum())
    N = int(len(y_true) - P)
    if P == 0 or N == 0:
        raise ValueError(
            "ROC undefined: need at least one positive and one negative sample"
        )

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)

    tpr = tp_cum / P
    fpr = fp_cum / N

    score_changes = np.r_[True, y_score_sorted[1:] != y_score_sorted[:-1]]
    fpr_pts = np.r_[0.0, fpr[score_changes], 1.0]
    tpr_pts = np.r_[0.0, tpr[score_changes], 1.0]
    thresholds = np.r_[np.inf, y_score_sorted[score_changes], -np.inf]

    auc_value = float(np.trapezoid(tpr_pts, fpr_pts))

    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr_pts, tpr_pts, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (positive={positive_group})  AUC={auc_value:.3f}")
    plt.tight_layout()

    if save:
        out_path = Path(out_path)  # type: ignore[arg-type]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")

    plt.close(fig)

    roc_df = pd.DataFrame({"fpr": fpr_pts, "tpr": tpr_pts, "threshold": thresholds})
    return roc_df, auc_value
