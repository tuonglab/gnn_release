import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV data
csv_path = "/scratch/project/tcr_ml/gnn_release/model_2025_hetero_isacs_only/graph_level_uncertainty/uncertainty_graph_level_pica_complete.csv"
df = pd.read_csv(csv_path)

# Define name of the custom output folder
output_subdir_name = "pica_complete"  # <-- change this as needed

# Define full output path
base_dir = os.path.dirname(csv_path)
plot_dir = os.path.join(base_dir, output_subdir_name)

# Create the folder if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)

# Set seaborn style
sns.set(style="whitegrid")

# Plot 1: Epistemic Uncertainty vs Predicted Probability
plt.figure(figsize=(14, 10))
sns.scatterplot(data=df, x="pred_prob_1", y="mutual_info", hue="sample_id", palette="tab10", legend=False)
plt.title("Epistemic Uncertainty vs Predicted Probability")
plt.xlabel("Predicted Probability (Class 1)")
plt.ylabel("Mutual Information")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "epistemic_vs_pred_prob.png"))
plt.show()

# Plot 2: Aleatoric Uncertainty vs Predicted Probability
plt.figure(figsize=(14, 10))
sns.scatterplot(data=df, x="pred_prob_1", y="aleatoric_var", hue="sample_id", palette="tab10", legend=False)
plt.title("Aleatoric Uncertainty vs Predicted Probability")
plt.xlabel("Predicted Probability (Class 1)")
plt.ylabel("Aleatoric Variance")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "aleatoric_vs_pred_prob.png"))
plt.show()

# Plot 3: Violin Plot - Aleatoric Uncertainty per Sample
plt.figure(figsize=(14, 10))
sns.violinplot(data=df, x="sample_id", y="aleatoric_var", inner="point")
plt.title("Aleatoric Uncertainty Distribution per Sample")
plt.xlabel("Sample ID")
plt.ylabel("Aleatoric Variance")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "aleatoric_per_sample.png"))
plt.show()

# Plot 4: Violin Plot - Epistemic Uncertainty per Sample
plt.figure(figsize=(14, 10))
sns.violinplot(data=df, x="sample_id", y="mutual_info", inner="point")
plt.title("Epistemic Uncertainty Distribution per Sample")
plt.xlabel("Sample ID")
plt.ylabel("Mutual Information")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "epistemic_per_sample.png"))
plt.show()
