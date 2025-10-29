from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Apply seaborn styling once at import
sns.set_theme(
    context="talk",
    style="whitegrid",
    font_scale=1.1,
    rc={"figure.dpi": 150}
)

def scatter_mean_scores(df_metrics: pd.DataFrame, out_path: str | Path) -> None:
    fig = plt.figure(figsize=(20, 10))
    sns.scatterplot(
        data=df_metrics,
        x=range(len(df_metrics["Mean Score"])),
        y="Mean Score",
        alpha=0.6
    )
    plt.title("Distribution of Mean Scores Across Files")
    plt.xlabel("File Index")
    plt.ylabel("Mean Score")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def boxplot_metrics(df_metrics: pd.DataFrame, metrics: list[str], out_path: str | Path) -> None:
    fig = plt.figure(figsize=(24, 10))
    sns.boxplot(data=df_metrics[metrics])
    plt.title("Boxplots of Cancer CDR3 Scores")
    plt.ylabel("Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def hist_sequence_scores(series_scores, out_path: str | Path, bins: int = 10) -> None:
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(
        series_scores.dropna(),
        bins=bins,
        kde=True,
        alpha=0.7
    )
    plt.title("Histogram of Individual Sequence Scores")
    plt.xlabel("Cancer Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

