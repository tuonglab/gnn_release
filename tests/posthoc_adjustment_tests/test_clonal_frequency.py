# test_add_row_frequencies.py

import pandas as pd
import pytest

from tcrgnn.posthoc_adjustment._clonal_frequency import add_row_frequencies


def write_model(tmp_path, rows):
    p = tmp_path / "model.csv"
    # no header, "sequence,score"
    pd.DataFrame(rows).to_csv(p, header=False, index=False)
    return p


@pytest.mark.parametrize("path_kind", ["str", "path"])
def test_happy_path_matches_sequences_and_preserves_order(tmp_path, path_kind):
    # model output with duplicate sequence to ensure repeated mapping works
    model_rows = [
        ["SEQ_A", 0.1],
        ["SEQ_B", 0.2],
        ["SEQ_A", 0.3],
    ]
    model_path = write_model(tmp_path, model_rows)
    model_arg = str(model_path) if path_kind == "str" else model_path

    # counts_df first col = sequence, second col = clonal freq, extra col should be ignored
    counts_df = pd.DataFrame(
        {
            "foo": ["SEQ_A", "SEQ_B", "SEQ_C"],
            "bar": [0.7, 0.3, 0.0],
            "extra": [1, 2, 3],
        }
    )

    out = add_row_frequencies(model_arg, counts_df)

    # expected columns exist
    assert {"sequence", "score", "clonal_frequency"}.issubset(out.columns)

    # order preserved and frequencies mapped by sequence
    assert out.loc[0, "sequence"] == "SEQ_A"
    assert out.loc[1, "sequence"] == "SEQ_B"
    assert out.loc[2, "sequence"] == "SEQ_A"

    # same mapped frequency for identical sequences
    assert out.loc[0, "clonal_frequency"] == 0.7
    assert out.loc[2, "clonal_frequency"] == 0.7
    assert out.loc[1, "clonal_frequency"] == 0.3

    # score column parsed as float
    assert out["score"].dtype.kind in ("f",)  # float dtype


def test_missing_frequency_raises_value_error_with_sequence_list(tmp_path):
    model_rows = [
        ["S1", 1.0],
        ["S2", 2.0],
    ]
    model_path = write_model(tmp_path, model_rows)

    # counts_df missing S2 frequency
    counts_df = pd.DataFrame(
        {
            "col1": ["S1"],
            "col2": [0.5],
        }
    )

    with pytest.raises(ValueError) as e:
        add_row_frequencies(model_path, counts_df)

    msg = str(e.value)
    # error message lists missing sequences
    assert "S2" in msg


def test_counts_df_with_different_column_names_and_extra_columns(tmp_path):
    model_rows = [
        ["X", 9.9],
        ["Y", 3.3],
    ]
    model_path = write_model(tmp_path, model_rows)

    # completely different names, plus an extra column
    counts_df = pd.DataFrame(
        {
            "seq_col": ["X", "Y"],
            "freq_col": [0.9, 0.1],
            "ignore_me": ["a", "b"],
        }
    )

    out = add_row_frequencies(model_path, counts_df)
    assert out.shape[0] == 2
    assert out.loc[out["sequence"] == "X", "clonal_frequency"].item() == 0.9
    assert out.loc[out["sequence"] == "Y", "clonal_frequency"].item() == 0.1


def test_duplicate_sequences_in_model_get_same_frequency(tmp_path):
    model_rows = [
        ["Z", 0.0],
        ["Z", 1.0],
        ["Z", 2.0],
    ]
    model_path = write_model(tmp_path, model_rows)

    counts_df = pd.DataFrame(
        {
            "first": ["Z"],
            "second": [0.42],
        }
    )

    add_row_frequencies(model_path, counts_df)
