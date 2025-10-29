from pathlib import Path
from .io import read_scores_csv, dataset_name, model_base_dir_from
from .prepare import prepare_labels_and_scores
from .charts import calculate_roc, plot_roc_curve, plot_boxplot

def compare_groups(cancer_csv: str, control_csv: str) -> dict:
    cancer_df = read_scores_csv(cancer_csv)
    control_df = read_scores_csv(control_csv)

    scores, labels, cancer_scores, control_scores = prepare_labels_and_scores(cancer_df, control_df)
    fpr, tpr, roc_auc, thresholds = calculate_roc(scores, labels)

    dataset = dataset_name(cancer_csv, control_csv)
    out_dir = model_base_dir_from(cancer_csv) / "roc_comparisons"

    roc_path = plot_roc_curve(fpr, tpr, roc_auc, out_dir, dataset)
    box_path = plot_boxplot(cancer_scores, control_scores, out_dir, dataset)

    optimal_idx = (tpr - fpr).argmax()
    return {
        "roc_auc": roc_auc,
        "roc_plot": roc_path,
        "box_plot": box_path,
        "optimal_threshold": thresholds[optimal_idx],
        "tpr": tpr[optimal_idx],
        "fpr": fpr[optimal_idx],
    }
