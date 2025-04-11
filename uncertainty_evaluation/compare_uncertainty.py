import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File paths (update these to your actual locations) ===
model_a_path = "/scratch/project/tcr_ml/gnn_release/model_2025_hetero_isacs_only/graph_level_uncertainty/uncertainty_graph_level_seekgene.csv"
model_b_path = "/scratch/project/tcr_ml/gnn_release/model_2025_hetero_isacs_ccdi/graph_level_uncertainty/uncertainty_graph_level_seekgene.csv"

# === Output directory for saving plots ===
output_dir = "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/plots"
os.makedirs(output_dir, exist_ok=True)

# === Load and label data ===
df_a = pd.read_csv(model_a_path)
df_b = pd.read_csv(model_b_path)

df_a["model"] = "ISACS Only"
df_b["model"] = "ISACS + CCDI"

# Set true label (e.g., cancer = 1 for aml_zero)
df_a["true_label"] = 1
df_b["true_label"] = 1

# Combine data
df = pd.concat([df_a, df_b], ignore_index=True)

# Add 'correct' column: 1 if correct prediction, else 0
df["correct"] = (df["pred_class"] == df["true_label"]).astype(int)

# === Plotting style ===
sns.set(style="whitegrid")

# === Plot 1: Violin - Epistemic Uncertainty vs Correctness ===
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="correct", y="mutual_info", hue="model", split=True)
plt.title("Epistemic Uncertainty by Prediction Correctness")
plt.xlabel("Correct Prediction (1 = Yes, 0 = No)")
plt.ylabel("Mutual Information")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "epistemic_uncertainty_vs_correctness.png"))
plt.close()

# === Plot 2: Violin - Aleatoric Uncertainty vs Correctness ===
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="correct", y="aleatoric_var", hue="model", split=True)
plt.title("Aleatoric Uncertainty by Prediction Correctness")
plt.xlabel("Correct Prediction (1 = Yes, 0 = No)")
plt.ylabel("Aleatoric Variance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "aleatoric_uncertainty_vs_correctness.png"))
plt.close()

# === Plot 3: Scatter - Mutual Info vs Predicted Prob, colored by correctness ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="pred_prob_1", y="mutual_info", hue="correct", style="model", alpha=0.6)
plt.title("Epistemic Uncertainty vs Confidence (Colored by Correctness)")
plt.xlabel("Predicted Probability (Class 1)")
plt.ylabel("Mutual Information")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "epistemic_vs_confidence_by_correctness.png"))
plt.close()

# === Plot 4: Scatter - Aleatoric Variance vs Predicted Prob, colored by correctness ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="pred_prob_1", y="aleatoric_var", hue="correct", style="model", alpha=0.6)
plt.title("Aleatoric Uncertainty vs Confidence (Colored by Correctness)")
plt.xlabel("Predicted Probability (Class 1)")
plt.ylabel("Aleatoric Variance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "aleatoric_vs_confidence_by_correctness.png"))
plt.close()
