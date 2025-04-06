import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def load_scores(cancer_file, control_file):
    try:
        cancer_df = pd.read_csv(cancer_file)
        control_df = pd.read_csv(control_file)

        if 'Mean Score' not in cancer_df.columns or 'Mean Score' not in control_df.columns:
            raise ValueError("Missing 'Mean Score' in one of the files.")

        scores = np.concatenate([
            cancer_df['Mean Score'].values,
            control_df['Mean Score'].values
        ])
        labels = np.concatenate([
            np.ones(len(cancer_df)),
            np.zeros(len(control_df))
        ])

        return scores, labels
    except Exception as e:
        print(f"Error loading data for {cancer_file}: {e}")
        raise

def calculate_roc(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc_multiple(models, dataset_folder, base_path, output_dir=None):
    dataset_name = dataset_folder.replace("_scores", "")
    plt.figure(figsize=(14, 10))  # Width=14 inches, Height=10 inches

    for model in models:
        model_path = os.path.join(base_path, model)
        cancer_file = os.path.join(model_path, dataset_folder, "metric_scores.csv")
        control_file = os.path.join(model_path, "control_scores", "metric_scores.csv")
        
        scores, labels = load_scores(cancer_file, control_file)
        fpr, tpr, roc_auc = calculate_roc(scores, labels)
        display_name = model_names.get(model, model)
        plt.plot(fpr, tpr, lw=2, label=f'{display_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison - {dataset_name}')
    plt.legend(loc="lower right", title="Models")

    # Set default output directory if none specified
    if output_dir is None:
        output_dir = os.path.join(base_path, models[0], "roc_comparisons")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'roc_comparison_{dataset_name}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ROC comparison plot to: {output_path}")


# === Streamlined input ===

models = ["model_2025_360_only", "model_2025_all_combo", "model_2025_ccdi_only","model_2025_isacs_360_no_ccdi","model_2025_isacs_ccdi","model_2025_isacs_only","model_2025_ccdi_360"]
dataset_folder = "non_360_timepoints_scores"
base_path = "/scratch/project/tcr_ml/gnn_release"
model_names = {
    "model_2025_360_only": "D-360",
    "model_2025_all_combo": "All",
    "model_2025_ccdi_only": "CCDI Only",
    "model_2025_isacs_360_no_ccdi": "ISACS + D-360",
    "model_2025_isacs_ccdi": "ISACS + CCDI",
    "model_2025_isacs_only": "ISACS",
    "model_2025_ccdi_360": "CCDI + D-360"
}

custom_output_dir = "/scratch/project/tcr_ml/gnn_release/model_comparisons_roc"
plot_roc_multiple(models, dataset_folder, base_path, output_dir=custom_output_dir)

