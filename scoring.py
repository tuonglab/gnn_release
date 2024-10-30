import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import kl_div
from sklearn.metrics import roc_auc_score, roc_curve

# Set global font size smaller for plots
plt.rcParams['font.size'] = 24  # General font size for the plot

def process_scores(directory) -> None:
    cancer_scores = []
    control_scores = []

    # Create a CSV file for the scores
    with open(f"{directory}/metric_scores.csv", "w", newline="") as scores_file:
        scores_writer = csv.writer(scores_file)

        # Write headers to the CSV file
        headers = [
            "Filename",
            "Normalized KL Divergence",
            "Hellinger Score",
            "Cosine Similarity",
            "Mean Score",
            "Total Variation Distance",
        ]
        scores_writer.writerow(headers)

        # Loop over all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                # Load data from text file (second column)
                scores = np.loadtxt(
                    os.path.join(directory, filename),
                    delimiter=",",
                    usecols=1,
                )

                # Normalize the scores to make them into probability distributions
                normalised_scores = scores / np.sum(scores)

                # Generate a perfect score distribution
                perfect_scores = np.ones_like(scores)
                normalised_perfect_scores = perfect_scores / np.sum(perfect_scores)

                # Calculate the KL Divergence
                kl_divergence = kl_div(normalised_scores, normalised_perfect_scores)
                overall_kl_divergence = np.sum(kl_divergence)
                normalized_kl_divergence = np.exp(-overall_kl_divergence)

                # Calculate Hellinger distance
                hellinger_score = 1 - np.sqrt(
                    0.5 * np.sum((np.sqrt(normalised_scores) - np.sqrt(normalised_perfect_scores)) ** 2)
                )

                # Calculate Cosine Similarity
                cosine_similarity = 1 - cosine(scores, perfect_scores)

                # Calculate Total Variation Distance
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
                    cancer_scores.append(
                        [
                            float(normalized_kl_divergence),
                            float(hellinger_score),
                            float(cosine_similarity),
                            float(np.mean(scores)),
                            float(tvd),
                        ]
                    )
                elif "control" in filename:
                    control_scores.append(
                        [
                            float(normalized_kl_divergence),
                            float(hellinger_score),
                            float(cosine_similarity),
                            float(np.mean(scores)),
                            float(tvd),
                        ]
                    )

    # Initialize a dictionary to store the results
    results = {}

    # Create a figure for plotting all ROC curves together
    plt.figure(figsize=(10, 8))

    for i, metric in enumerate(
        [
            "Normalized KL Divergence",
            "Hellinger Distance",
            "Cosine Similarity",
            "Mean Scores",
            "Total Variation Distance",
        ]
    ):
        y_true = [1] * len(cancer_scores) + [0] * len(control_scores)
        y_scores = [score[i] for score in cancer_scores + control_scores]
        auc_score = roc_auc_score(y_true, y_scores)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        # Plot ROC curve on the same figure
        plt.plot(fpr, tpr, label=f"{metric} (AUC = {auc_score:.2f})")

    # Plot settings
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Set the new title
    plt.title("Cancer vs Control Samples (Hyperparameter Optimised Model)")

    # Customize the legend to be smaller and repositioned if necessary
    plt.legend(loc="lower right", fontsize=14)  # Reduce legend font size
    # Optionally, move the legend outside of the plot:
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14)

    # Save the combined ROC curve as a PDF
    plt.savefig(f"{directory}/combined_roc_curves.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# Example usage
directory = "/scratch/project/tcr_ml/gnn_release/model/scores"
process_scores(directory)
