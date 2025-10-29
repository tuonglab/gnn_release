import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === Illustrator-friendly fonts ===
matplotlib.rcParams["pdf.fonttype"] = 42  # embed fonts as TrueType
matplotlib.rcParams["ps.fonttype"] = 42

# === Define models (name -> base directory) ===
# Update these paths to your actual model output roots
models = {
    "Model Bulk + Single-Cell": "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/model_2025_uncertainty_curated",
    "Model Bulk Only": "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/model_2025_uncertainty_bulk",
    "Model Single-Cell Only": "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/model_2025_uncertainty_sc",
}

# === Dataset filenames and ground truth labels ===
# Each dataset is its own file name, which is the same across model dirs
dataset_files = {
    "AML ZERO": ("uncertainty_graph_level_aml_zero.csv", 1),
    "PICA COMPLETE": ("uncertainty_graph_level_pica_complete.csv", 0),
    "PHS002517": ("uncertainty_graph_level_pica_complete.csv", 1),
    "SARCOMA ZERO": ("uncertainty_graph_level_sarcoma_zero.csv", 1),
}

# === Output directory root ===
output_root = (
    "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/model_comparisons"
)
os.makedirs(output_root, exist_ok=True)


# === Helper to sanitize names for filenames ===
def safe_name(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_")


# === Load data for a given dataset across all models ===
def load_dataset_across_models(dataset_filename: str, true_label: int):
    data_by_model = {}
    for model_name, model_dir in models.items():
        csv_path = os.path.join(model_dir, "graph_level_uncertainty", dataset_filename)
        if not os.path.exists(csv_path):
            # If a model is missing this dataset file, skip it but keep comment handy
            # print(f"Missing file for {model_name}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["true_label"] = true_label
        df["correct"] = (df["pred_class"] == df["true_label"]).astype(int)
        data_by_model[model_name] = df
    return data_by_model


# === Plotting: for a single dataset, make a 3-panel figure per uncertainty type ===
def plot_dataset_comparison(
    dataset_name: str, data_by_model: dict, metric: str, ylabel: str
):
    # Ensure stable model order based on the models dict above
    ordered_models = [m for m in models.keys() if m in data_by_model]
    if not ordered_models:
        return None

    n_models = len(ordered_models)
    # 3 models -> width ~ 6 per panel, height ~ 7 looks nice for violins
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7), sharey=True)

    # When n_models == 1, axes is a single Axes
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, ordered_models):
        df = data_by_model[model_name]
        sns.violinplot(ax=ax, data=df, x="correct", y=metric, inner="box")
        ax.set_title(model_name, fontsize=14)
        ax.set_xlabel("Correct Prediction\n(1 = Yes, 0 = No)", fontsize=12)
        ax.set_ylabel(ylabel if ax is axes[0] else "", fontsize=12)

    fig.suptitle(
        f"{ylabel} by Prediction Correctness\n{dataset_name}",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# === Main loop: one PDF per dataset per uncertainty type ===
for dataset_name, (filename, true_label) in dataset_files.items():
    # Load all model outputs for this dataset
    data_by_model = load_dataset_across_models(filename, true_label)
    if not data_by_model:
        # print(f"No data loaded for {dataset_name}. Skipping.")
        continue

    # Dataset-specific output directory
    ds_out_dir = os.path.join(output_root, safe_name(dataset_name))
    os.makedirs(ds_out_dir, exist_ok=True)

    # 1) Epistemic: mutual_info
    fig_epi = plot_dataset_comparison(
        dataset_name=dataset_name,
        data_by_model=data_by_model,
        metric="mutual_info",
        ylabel="Mutual Information",
    )
    if fig_epi is not None:
        out_path_epi = os.path.join(
            ds_out_dir, f"{safe_name(dataset_name)}_epistemic_comparison.pdf"
        )
        fig_epi.savefig(out_path_epi, format="pdf")
        plt.close(fig_epi)

    # 2) Aleatoric: aleatoric_var
    fig_alea = plot_dataset_comparison(
        dataset_name=dataset_name,
        data_by_model=data_by_model,
        metric="aleatoric_var",
        ylabel="Aleatoric Variance",
    )
    if fig_alea is not None:
        out_path_alea = os.path.join(
            ds_out_dir, f"{safe_name(dataset_name)}_aleatoric_comparison.pdf"
        )
        fig_alea.savefig(out_path_alea, format="pdf")
        plt.close(fig_alea)
