import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

gnn_no_clonal_weightage = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/model_2025_sc_curated/seekgene_scores/metric_scores.csv"
)
gnn_clonal_weightage = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/icantcrscoring/model_2025_sc_curated/seekgene/summary.csv"
)
icantcr_scoring = pd.read_csv(
    "/scratch/project/tcr_ml/iCanTCR/output/seekgene/summary_seekgene_result.csv"
)

scores = {
    "GNN (no clonal weightage)": gnn_no_clonal_weightage["Mean Score"],
    "GNN (clonal weightage)": gnn_clonal_weightage["weight_score"],
    "iCanTCR scoring": icantcr_scoring["Overall_Score"],
}


dataset = "seekgene"

plt.figure(figsize=(16, 10))
plt.boxplot(scores.values(), labels=scores.keys())
plt.ylabel("Scores")
plt.title(dataset)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"model_scores_boxplot_{dataset.lower()}.pdf", format="pdf", dpi=600)

gnn_no_clonal_weightage_pica = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/model_2025_sc_curated/PICA_scores/metric_scores.csv"
)
gnn_clonal_weightage_pica = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/icantcrscoring/model_2025_sc_curated/PICA/summary.csv"
)
icantcr_scoring_pica = pd.read_csv(
    "/scratch/project/tcr_ml/iCanTCR/output/PICA_GLIPH/accumulated_pica_summary.csv"
)

scores_pica = {
    "GNN (no clonal weightage)": gnn_no_clonal_weightage_pica["Mean Score"],
    "GNN (clonal weightage)": gnn_clonal_weightage_pica["weight_score"],
    "iCanTCR scoring": icantcr_scoring_pica["Overall_Score"],
}

dataset_control = "PICA"

plt.figure(figsize=(16, 10))
plt.boxplot(scores_pica.values(), labels=scores_pica.keys())
plt.ylabel("Scores")
plt.title(dataset_control)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(
    f"model_scores_boxplot_{dataset_control.lower()}.pdf", format="pdf", dpi=600
)


def plot_roc_between_groups(models, cancer_name, control_name):
    """
    models: dict with keys = model names,
            values = (cancer_scores_array, control_scores_array)
    cancer_name: str name of the cancer dataset
    control_name: str name of the control dataset
    """

    def combine_scores(cancer_scores, control_scores):
        combined = np.concatenate([cancer_scores, control_scores])
        labels = np.concatenate(
            [np.ones(len(cancer_scores)), np.zeros(len(control_scores))]
        )
        return combined, labels

    plt.figure(figsize=(16, 10))
    for name, (cancer_scores, control_scores) in models.items():
        y_scores, y_true = combine_scores(cancer_scores, control_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {cancer_name} (cancer) vs {control_name} (control)")
    plt.legend(loc="lower right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    out_name = f"roc_curve_{cancer_name.lower()}_vs_{control_name.lower()}.pdf"
    plt.savefig(out_name, format="pdf", dpi=600)
    plt.close()
    print(f"Saved ROC curve as {out_name}")


# Example usage
models = {
    "GNN (no clonal weightage)": (
        gnn_no_clonal_weightage["Mean Score"].values,
        gnn_no_clonal_weightage_pica["Mean Score"].values,
    ),
    "GNN (clonal weightage)": (
        gnn_clonal_weightage["weight_score"].values,
        gnn_clonal_weightage_pica["weight_score"].values,
    ),
    "iCanTCR": (
        icantcr_scoring["Overall_Score"].values,
        icantcr_scoring_pica["Overall_Score"].values,
    ),
}

plot_roc_between_groups(models, cancer_name="seekgene", control_name="PICA")
