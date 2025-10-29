from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_txt(directory: str | Path) -> list[Path]:
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.suffix == ".txt")


def read_score_txt(path: str | Path) -> pd.DataFrame:
    # columns: CDR3_Sequence, Cancer_Score
    return pd.read_csv(
        path, sep=",", header=None, names=["CDR3_Sequence", "Cancer_Score"]
    )


def read_metrics_csv(
    directory: str | Path, filename: str = "metric_scores.csv"
) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / filename)


def ensure_outdir(directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_scores_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Mean Score" not in df:
        raise ValueError(f"Missing Mean Score in {path}")
    return df


def model_base_dir_from(cancer_csv: str | Path) -> Path:
    # go up two directories from cancer csv, same as before
    p = Path(cancer_csv).resolve()
    return p.parents[1]


def dataset_name(cancer_csv: str | Path, control_csv: str | Path) -> str:
    c1 = Path(cancer_csv).parent.name.replace("_scores", "")
    c0 = Path(control_csv).parent.name.replace("_scores", "")
    return f"{c1}_vs_{c0}"
