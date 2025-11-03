import pytest

from tcrgnn.posthoc_adjustment._clonal_frequency import add_row_frequencies


def test_add_row_frequencies_computes_expected_values(tmp_path):
    csv_path = tmp_path / "model_output.txt"
    csv_path.write_text("seq1,0.1\nseq2,0.2\nseq1,0.3\n", encoding="utf-8")
    counts = [2, 3, 5]

    df = add_row_frequencies(str(csv_path), counts)

    assert list(df["count"]) == counts
    assert list(df["clonal_frequency"]) == pytest.approx([0.7, 0.3, 0.7])


def test_add_row_frequencies_raises_on_count_length_mismatch(tmp_path):
    csv_path = tmp_path / "model_output.txt"
    csv_path.write_text("seq1,0.1\nseq2,0.2\n", encoding="utf-8")
    counts = [1]

    with pytest.raises(
        ValueError, match="Length of counts does not match number of sequences"
    ):
        add_row_frequencies(str(csv_path), counts)


def test_add_row_frequencies_raises_on_non_positive_total(tmp_path):
    csv_path = tmp_path / "model_output.txt"
    csv_path.write_text("seq1,0.1\nseq2,0.2\n", encoding="utf-8")
    counts = [0, 0]

    with pytest.raises(ValueError, match="Total counts must be positive"):
        add_row_frequencies(str(csv_path), counts)
