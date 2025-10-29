import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

# Load the two files
file1 = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/icantcrscoring/seekgene/summary.csv"
)
file2 = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/icantcrscoring/pica/summary.csv"
)

# Extract weight scores
scores_cancer = file1["weight_score"].dropna()
scores_control = file2["weight_score"].dropna()

# 1️⃣ Boxplot

plt.figure(figsize=(15, 10))
plt.boxplot(
    [scores_cancer, scores_control], labels=["Seekgene (Cancer)", "PICA (Control)"]
)
plt.ylabel("Weight Score")
plt.title("Boxplot of Weight Scores")
plt.grid(axis="y")
plt.savefig("boxplot_weight_scores.png")
plt.show()

# 2️⃣ ROC curve

y_true = [1] * len(scores_cancer) + [0] * len(scores_control)
y_scores = list(scores_cancer) + list(scores_control)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(15, 10))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Seekgene (Cancer) vs PICA (Control) Weight Scores")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("roc_curve_weight_scores.png")
plt.show()
