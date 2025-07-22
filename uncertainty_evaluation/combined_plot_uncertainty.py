import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os

# === Set fonts for Illustrator compatibility ===
matplotlib.rcParams["pdf.fonttype"] = 42  # embed fonts as TrueType
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

# === Define model name and base directory ===
model_name = "Model Bulk + Single Cell"
model_dir = "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/model_2025_uncertainty_curated"

# === Define dataset filenames and corresponding true labels ===
dataset_files = {
    "AML ZERO": ("uncertainty_graph_level_aml_zero.csv", 1),
    "PICA COMPLETE": ("uncertainty_graph_level_pica_complete.csv", 0),
    "PHS002517": ("uncertainty_graph_level_pica_complete.csv", 1),
    "SARCOMA ZERO": ("uncertainty_graph_level_sarcoma_zero.csv", 1)
}

# === Output directory ===
output_dir = "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/model_2025_uncertainty_curated/plots"
os.makedirs(output_dir, exist_ok=True)

# === Load and prepare data ===
dfs = {}
for dataset_name, (filename, true_label) in dataset_files.items():
    full_path = os.path.join(model_dir, "graph_level_uncertainty", filename)
    df = pd.read_csv(full_path)
    df["true_label"] = true_label
    df["correct"] = (df["pred_class"] == df["true_label"]).astype(int)
    dfs[dataset_name] = df


# === Plot 1: Epistemic Uncertainty ===
fig, axes = plt.subplots(1, 4, figsize=(24, 8), sharey=True)
for ax, (dataset_name, df) in zip(axes, dfs.items()):
    sns.violinplot(ax=ax, data=df, x="correct", y="mutual_info", inner="point")
    ax.set_title(dataset_name, fontsize=16)
    ax.set_xlabel("Correct Prediction\n(1 = Yes, 0 = No)", fontsize=14)
    ax.set_ylabel("Mutual Information" if ax == axes[0] else "", fontsize=14)

fig.suptitle(f"Epistemic Uncertainty by Prediction Correctness\n({model_name})", fontsize=24, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
pdf_path = os.path.join(output_dir, f"epistemic_uncertainty_{model_name.lower().replace(' ', '_')}.pdf")
plt.savefig(pdf_path, format="pdf")
plt.close()

# === Plot 2: Aleatoric Uncertainty ===
fig, axes = plt.subplots(1, 4, figsize=(24, 8), sharey=True)
for ax, (dataset_name, df) in zip(axes, dfs.items()):
    sns.violinplot(ax=ax, data=df, x="correct", y="aleatoric_var", inner="point")
    ax.set_title(dataset_name, fontsize=16)
    ax.set_xlabel("Correct Prediction\n(1 = Yes, 0 = No)", fontsize=14)
    ax.set_ylabel("Aleatoric Variance" if ax == axes[0] else "", fontsize=14)

fig.suptitle(f"Aleatoric Uncertainty by Prediction Correctness\n({model_name})", fontsize=24, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
pdf_path = os.path.join(output_dir, f"aleatoric_uncertainty_{model_name.lower().replace(' ', '_')}.pdf")
plt.savefig(pdf_path, format="pdf")
plt.close()
