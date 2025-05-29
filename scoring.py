import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import kl_div
from sklearn.metrics import roc_auc_score, roc_curve

# Set global font size smaller for plots
plt.rcParams['font.size'] = 24  # General font size for the plot

def process_scores(directory: str) -> None:
    cancer_scores = []
    control_scores = []

    # Create a CSV file for the scores
    with open(f"{directory}/metric_scores.csv", "w", newline="") as scores_file:
        scores_writer = csv.writer(scores_file)

        headers = [
            "Filename",
            "Normalized KL Divergence",
            "Hellinger Score",
            "Cosine Similarity",
            "Mean Score",
            "Total Variation Distance",
        ]
        scores_writer.writerow(headers)

        for filename in os.listdir(directory):
            print(filename)
            if filename.endswith(".txt"):
                scores = np.genfromtxt(
                    os.path.join(directory, filename),
                    delimiter=",",
                    usecols=1,
                    dtype=float,
                    encoding="utf-8",
                    invalid_raise=False
                )
                scores = scores[~np.isnan(scores)]
                normalised_scores = scores / np.sum(scores)

                perfect_scores = np.ones_like(scores)
                normalised_perfect_scores = perfect_scores / np.sum(perfect_scores)

                kl_divergence = kl_div(normalised_scores, normalised_perfect_scores)
                overall_kl_divergence = np.sum(kl_divergence)
                normalized_kl_divergence = np.exp(-overall_kl_divergence)

                hellinger_score = 1 - np.sqrt(
                    0.5 * np.sum((np.sqrt(normalised_scores) - np.sqrt(normalised_perfect_scores)) ** 2)
                )

                cosine_similarity = 1 - cosine(scores, perfect_scores)
                tvd = 1 - (0.5 * np.sum(np.abs(normalised_scores - normalised_perfect_scores)))

                scores_list = [
                    filename,
                    float(normalized_kl_divergence),
                    float(hellinger_score),
                    float(cosine_similarity),
                    float(np.mean(scores)),
                    float(tvd),
                ]
                scores_writer.writerow(scores_list)

                if "cancer" in filename:
                    cancer_scores.append(scores_list[1:])
                elif "control" in filename:
                    control_scores.append(scores_list[1:])

    plt.figure(figsize=(10, 8))
    for i, metric in enumerate([
        "Normalized KL Divergence",
        "Hellinger Distance",
        "Cosine Similarity",
        "Mean Scores",
        "Total Variation Distance",
    ]):
        y_true = [1] * len(cancer_scores) + [0] * len(control_scores)
        y_scores = [score[i] for score in cancer_scores + control_scores]
        auc_score = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f"{metric} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cancer vs Control Samples (Hyperparameter Optimised Model)")
    plt.legend(loc="lower right", fontsize=14)
    plt.savefig(f"{directory}/combined_roc_curves.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process score files and generate ROC plots.")
    parser.add_argument(
        "--directory",
        type=str,
        default="/scratch/project/tcr_ml/gnn_release/model_2025_ccdi_only/pica_filtered_scores",
        help="Directory containing the score .txt files. Defaults to a preset path.",
    )
    args = parser.parse_args()
    process_scores(args.directory)
