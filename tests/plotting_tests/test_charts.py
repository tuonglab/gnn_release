import matplotlib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tcrgnn.plotting import charts

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    charts.plt.close("all")
    yield
    charts.plt.close("all")


def test_boxplot_individual_sample_saves_and_closes(tmp_path, monkeypatch):
    saved_paths = []

    def fake_savefig(path, *args, **kwargs):
        saved_paths.append(path)

    monkeypatch.setattr(charts.plt, "savefig", fake_savefig)
    out_path = tmp_path / "box.png"

    charts.boxplot_individual_sample([0.1, 0.5, 0.9], save=True, out_path=out_path)

    assert saved_paths == [out_path]
    assert charts.plt.get_fignums() == []


def test_scatterplot_individual_sample_saves_and_closes(tmp_path, monkeypatch):
    saved_paths = []

    def fake_savefig(path, *args, **kwargs):
        saved_paths.append(path)

    monkeypatch.setattr(charts.plt, "savefig", fake_savefig)
    out_path = tmp_path / "scatter.png"

    charts.scatterplot_individual_sample([0.2, 0.4, 0.6], save=True, out_path=out_path)

    assert saved_paths == [out_path]
    assert charts.plt.get_fignums() == []


def test_plot_inv_logit_per_source_summary_and_saves(tmp_path, monkeypatch):
    saved_paths = []

    def fake_savefig(path, *args, **kwargs):
        saved_paths.append(path)

    monkeypatch.setattr(charts.plt, "savefig", fake_savefig)

    df = pd.DataFrame(
        {
            "sequence": ["s1", "s2", "s3"],
            "scores": [0.0, 2.0, -2.0],
            "source": ["A", "A", "B"],
        }
    )
    out_dir = tmp_path / "plots"
    summary = charts.plot_inv_logit_per_source(df, save=True, out_dir=out_dir)

    expected_values = {
        "A": float(np.mean(charts.expit(np.array([0.0, 2.0], dtype=float)))),
        "B": float(charts.expit(-2.0)),
    }

    assert out_dir.is_dir()
    assert len(saved_paths) == 2
    assert {p.name for p in saved_paths} == {
        "inv_logit_boxplot.png",
        "inv_logit_mean_scatterplot.png",
    }
    assert summary["source"].tolist() == ["A", "B"]
    for source, expected in expected_values.items():
        assert summary.set_index("source").loc[
            source, "inv_logit_mean"
        ] == pytest.approx(expected)
    assert charts.plt.get_fignums() == []


def test_summarize_and_plot_inv_logit_means_outputs_and_save(tmp_path, monkeypatch):
    saved_paths = []

    def fake_savefig(path, *args, **kwargs):
        saved_paths.append(path)

    monkeypatch.setattr(charts.plt, "savefig", fake_savefig)

    cancer_df = pd.DataFrame(
        {
            "sequence": ["c1", "c2", "c3"],
            "scores": [0.0, 1.0, -1.0],
            "source": ["A", "A", "B"],
        }
    )
    control_df = pd.DataFrame(
        {
            "sequence": ["n1", "n2", "n3"],
            "scores": [-0.5, 0.5, 1.5],
            "source": ["A", "B", "B"],
        }
    )
    out_dir = tmp_path / "summary_plots"
    summary_long, summary_wide = charts.summarize_and_plot_inv_logit_means(
        cancer_df, control_df, save=True, out_dir=out_dir
    )

    expected_long = pd.DataFrame(
        [
            {
                "source": "A",
                "group": "Cancer",
                "inv_logit_mean": float(
                    np.mean(charts.expit(np.array([0.0, 1.0], dtype=float)))
                ),
            },
            {
                "source": "B",
                "group": "Cancer",
                "inv_logit_mean": float(charts.expit(-1.0)),
            },
            {
                "source": "B",
                "group": "Control",
                "inv_logit_mean": float(
                    np.mean(charts.expit(np.array([0.5, 1.5], dtype=float)))
                ),
            },
            {
                "source": "A",
                "group": "Control",
                "inv_logit_mean": float(charts.expit(-0.5)),
            },
        ]
    )
    expected_wide = pd.DataFrame(
        [
            {
                "source": "A",
                "Cancer": expected_long.iloc[0]["inv_logit_mean"],
                "Control": expected_long.iloc[3]["inv_logit_mean"],
            },
            {
                "source": "B",
                "Cancer": expected_long.iloc[1]["inv_logit_mean"],
                "Control": expected_long.iloc[2]["inv_logit_mean"],
            },
        ]
    )

    assert out_dir.is_dir()
    assert saved_paths == [
        out_dir / "inv_logit_mean_cancer_vs_control_boxplot.png",
    ]
    assert_frame_equal(
        summary_long.reset_index(drop=True),
        expected_long,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
    assert_frame_equal(
        summary_wide.sort_values("source").reset_index(drop=True),
        expected_wide,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
    assert charts.plt.get_fignums() == []


def test_plot_roc_from_summary_computes_auc_and_saves(tmp_path, monkeypatch):
    saved_paths = []

    def fake_savefig(path, *args, **kwargs):
        saved_paths.append(path)

    monkeypatch.setattr(charts.plt, "savefig", fake_savefig)

    summary = pd.DataFrame(
        {
            "source": ["S1", "S2", "S3", "S4"],
            "group": ["Cancer", "Control", "Cancer", "Control"],
            "inv_logit_mean": [0.9, 0.8, 0.2, 0.1],
        }
    )
    out_path = tmp_path / "roc" / "curve.png"

    roc_df, auc_value = charts.plot_roc_from_summary(
        summary, save=True, out_path=out_path
    )

    assert out_path.parent.is_dir()
    assert saved_paths == [out_path]
    assert list(roc_df.columns) == ["fpr", "tpr", "threshold"]
    assert auc_value == pytest.approx(0.75, rel=1e-6)
    assert roc_df.iloc[0]["fpr"] == 0.0
    assert roc_df.iloc[0]["tpr"] == 0.0
    assert np.isposinf(roc_df.iloc[0]["threshold"])
    assert roc_df.iloc[-1]["fpr"] == 1.0
    assert roc_df.iloc[-1]["tpr"] == 1.0
    assert np.isneginf(roc_df.iloc[-1]["threshold"])
    assert charts.plt.get_fignums() == []


def test_plot_roc_from_summary_requires_both_classes():
    data = pd.DataFrame({"group": ["Cancer", "Cancer"], "inv_logit_mean": [0.2, 0.4]})
    with pytest.raises(ValueError, match="need at least one positive and one negative"):
        charts.plot_roc_from_summary(data)


def test_plot_roc_from_summary_missing_columns():
    data = pd.DataFrame({"group": ["Cancer"], "score": [0.1]})
    with pytest.raises(
        ValueError, match="summary_long must contain 'group' and 'inv_logit_mean'"
    ):
        charts.plot_roc_from_summary(data)


def test_plot_roc_from_summary_empty_after_nan_drop():
    df = pd.DataFrame(
        {"group": ["Cancer", "Control"], "inv_logit_mean": [np.nan, np.nan]}
    )
    with pytest.raises(ValueError, match="No data after dropping NaNs"):
        charts.plot_roc_from_summary(df)


def test_plot_roc_from_summary_unknown_positive_group():
    df = pd.DataFrame({"group": ["Control", "Control"], "inv_logit_mean": [0.2, 0.8]})
    # zero Cancer samples
    with pytest.raises(ValueError, match="need at least one positive and one negative"):
        charts.plot_roc_from_summary(df)


def test_plot_roc_from_summary_missing_score_column():
    df = pd.DataFrame({"group": ["Cancer", "Control"], "not_score": [0.2, 0.8]})
    with pytest.raises(
        ValueError, match="summary_long must contain 'group' and 'inv_logit_mean'"
    ):
        charts.plot_roc_from_summary(df)


def test_plot_roc_from_summary_missing_group_column():
    df = pd.DataFrame({"inv_logit_mean": [0.2, 0.8], "other": ["x", "y"]})
    with pytest.raises(
        ValueError, match="summary_long must contain 'group' and 'inv_logit_mean'"
    ):
        charts.plot_roc_from_summary(df)


def test_plot_roc_from_summary_all_nan_group_column():
    df = pd.DataFrame({"group": [np.nan, np.nan], "inv_logit_mean": [0.3, 0.7]})
    with pytest.raises(ValueError, match="No data after dropping NaNs"):
        charts.plot_roc_from_summary(df)


def test_plot_inv_logit_per_source_missing_sequence():
    df = pd.DataFrame({"scores": [1], "source": ["A"]})
    with pytest.raises(ValueError, match="Input DataFrame must contain columns"):
        charts.plot_inv_logit_per_source(df)


def test_summarize_missing_column_in_cancer_df():
    cancer_df = pd.DataFrame({"scores": [1], "source": ["A"]})
    control_df = pd.DataFrame({"sequence": ["x"], "scores": [1], "source": ["A"]})
    with pytest.raises(ValueError, match="cancer_df must contain columns"):
        charts.summarize_and_plot_inv_logit_means(cancer_df, control_df)
