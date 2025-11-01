from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit  # inverse-logit

# Apply seaborn styling once at import
sns.set_theme(context="talk", style="whitegrid", font_scale=1.1, rc={"figure.dpi": 150})


def boxplot_individual_sample(
    scores: np.ndarray | list[float], save: bool = False, out_path: str | Path = None
) -> None:
    fig = plt.figure(figsize=(24, 10))
    sns.boxplot(data=scores)
    plt.title("Boxplots of Individual Sample Scores")
    plt.ylabel("Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save and out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def scatterplot_individual_sample(
    scores: np.ndarray | list[float],
    save: bool = False,
    out_path: str | Path = None,
) -> None:
    """
    Scatterplot of individual sample scores.
    """
    fig = plt.figure(figsize=(24, 10))
    sns.scatterplot(x=np.arange(len(scores)), y=scores, s=80)
    plt.title("Scatterplot of Individual Sample Scores")
    plt.ylabel("Score")
    plt.xlabel("Index")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save and out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")

    plt.close(fig)


def plot_inv_logit_per_source(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: str | Path = None,
) -> pd.DataFrame:
    """
    Given dataframe with columns: sequence, scores, source,
    compute inverse-logit mean for each source, then plot
    boxplot + scatterplot over sources.

    Returns: summary DataFrame with columns:
        source, inv_logit_mean
    """

    # basic validation
    required_cols = {"sequence", "scores", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # compute inverse-logit per row
    df = df.copy()
    df["inv_logit"] = expit(df["scores"].astype(float))

    # group / aggregate
    summary = (
        df.groupby("source")["inv_logit"]
        .mean()
        .reset_index()
        .rename(columns={"inv_logit": "inv_logit_mean"})
    )

    # ensure deterministic ordering
    summary = summary.sort_values("inv_logit_mean", ascending=False).reset_index(
        drop=True
    )

    # create output directory if saving
    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # BOXPLOT
    # -----------------------------
    fig1 = plt.figure(figsize=(24, 10))
    sns.boxplot(data=df, x="source", y="inv_logit")
    plt.title("Inverse-Logit Score Distribution per Source")
    plt.ylabel("Inverse-logit score")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save and out_dir is not None:
        plt.savefig(out_dir / "inv_logit_boxplot.png", bbox_inches="tight")

    plt.close(fig1)

    # -----------------------------
    # SCATTERPLOT (summary means)
    # -----------------------------
    fig2 = plt.figure(figsize=(24, 10))
    sns.scatterplot(
        data=summary,
        x="source",
        y="inv_logit_mean",
        s=200,
    )
    plt.title("Inverse-Logit Mean per Source")
    plt.ylabel("Inverse-logit mean")
    plt.xlabel("Source")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save and out_dir is not None:
        plt.savefig(out_dir / "inv_logit_mean_scatterplot.png", bbox_inches="tight")

    plt.close(fig2)

    return summary


def summarize_and_plot_inv_logit_means(
    cancer_df: pd.DataFrame,
    control_df: pd.DataFrame,
    save: bool = False,
    out_dir: str | Path = None,
):
    """
    Compute inverse-logit mean per source for cancer/control, plot a boxplot
    comparing the distributions of per-source means, and return summary frames.

    Parameters
    ----------
    cancer_df : pd.DataFrame  (columns: sequence, scores, source)
    control_df : pd.DataFrame (columns: sequence, scores, source)
    save : bool
        If True, save plot to out_dir.
    out_dir : str | Path | None
        Directory to save plot(s) into.

    Returns
    -------
    summary_long : pd.DataFrame
        Columns: source, group, inv_logit_mean
        One row per source per group.
    summary_wide : pd.DataFrame
        Columns: source, Cancer, Control (wide pivot).
        Useful for downstream ROC between Cancer and Control per source.
    """
    required = {"sequence", "scores", "source"}
    if not required.issubset(cancer_df.columns):
        raise ValueError(f"cancer_df must contain columns: {required}")
    if not required.issubset(control_df.columns):
        raise ValueError(f"control_df must contain columns: {required}")

    # Copy to avoid mutating caller data
    c_df = cancer_df.copy()
    n_df = control_df.copy()

    # Inverse-logit per row
    c_df["inv_logit"] = expit(c_df["scores"].astype(float))
    n_df["inv_logit"] = expit(n_df["scores"].astype(float))

    # Aggregate per source
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

    # Combine
    summary_long = (
        pd.concat([c_summary, n_summary], ignore_index=True)
        .loc[:, ["source", "group", "inv_logit_mean"]]
        .sort_values(["group", "inv_logit_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Wide version (useful for ROC: align per-source Cancer vs Control)
    summary_wide = (
        summary_long.pivot_table(
            index="source", columns="group", values="inv_logit_mean", aggfunc="first"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Plot: boxplot of per-source means by group (two boxes)
    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))
    sns.boxplot(data=summary_long, x="group", y="inv_logit_mean")
    plt.title("Per-Source Inverse-Logit Means: Cancer vs Control")
    plt.ylabel("Inverse-logit mean (per source)")
    plt.xlabel("")
    plt.tight_layout()

    if save and out_dir is not None:
        plt.savefig(
            out_dir / "inv_logit_mean_cancer_vs_control_boxplot.png",
            bbox_inches="tight",
        )
    plt.close(fig)

    return summary_long, summary_wide


def plot_roc_from_summary(
    summary_long: pd.DataFrame,
    positive_group: str = "Cancer",
    score_col: str = "inv_logit_mean",
    group_col: str = "group",
    save: bool = False,
    out_path: str | Path | None = None,
):
    """
    Plot ROC using per-source scores from `summary_long` and return the ROC table and AUC.

    Parameters
    ----------
    summary_long : pd.DataFrame
        Must contain at least [group_col, score_col]. Typical from previous step:
          columns: ["source", "group", "inv_logit_mean"]
        Each row is a sample (e.g., one source in Cancer or Control) with a score.
    positive_group : str
        Label in `group_col` to treat as positive class.
    score_col : str
        Column with continuous scores.
    group_col : str
        Column with class labels.
    save : bool
        If True and out_path is provided, save the ROC figure.
    out_path : str | Path | None
        Path to save figure.

    Returns
    -------
    roc_df : pd.DataFrame
        Columns: fpr, tpr, threshold. Sorted by fpr ascending.
    auc_value : float
        Area under the ROC curve in [0, 1].
    """
    if group_col not in summary_long.columns or score_col not in summary_long.columns:
        raise ValueError(f"summary_long must contain '{group_col}' and '{score_col}'")

    df = summary_long[[group_col, score_col]].dropna().copy()
    if df.empty:
        raise ValueError("No data after dropping NaNs")

    y_true = (df[group_col].astype(str).values == str(positive_group)).astype(int)
    y_score = df[score_col].astype(float).values

    P = y_true.sum()
    N = len(y_true) - P
    if P == 0 or N == 0:
        raise ValueError(
            "ROC undefined: need at least one positive and one negative sample"
        )

    # Sort by score descending
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # Cumulative sums to build the stepwise ROC
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)

    tpr = tp_cum / P
    fpr = fp_cum / N

    # Keep points only where score changes, plus final point
    # This compresses flat segments for a clean ROC
    score_changes = np.r_[True, y_score_sorted[1:] != y_score_sorted[:-1]]
    fpr_pts = np.r_[0.0, fpr[score_changes], 1.0]
    tpr_pts = np.r_[0.0, tpr[score_changes], 1.0]
    thresholds = np.r_[np.inf, y_score_sorted[score_changes], -np.inf]

    # AUC via trapezoid rule
    auc_value = float(np.trapz(tpr_pts, fpr_pts))

    # Plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr_pts, tpr_pts, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (positive={positive_group})  AUC={auc_value:.3f}")
    plt.tight_layout()

    if save and out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")

    plt.close(fig)

    roc_df = pd.DataFrame({"fpr": fpr_pts, "tpr": tpr_pts, "threshold": thresholds})
    return roc_df, auc_value
