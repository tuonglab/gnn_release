import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def identify_outliers_with_files(data, column, directory):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    print(f"\nOutliers for {column}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Lower bound: {lower_bound:.4f}")
    print(f"Upper bound: {upper_bound:.4f}")

    all_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    print("\nOutlier values and their source files:")
    for score in outliers[column].sort_values(ascending=False):
        files_with_score = []
        for filename in all_files:
            file_path = os.path.join(directory, filename)
            try:
                file_data = pd.read_csv(
                    file_path,
                    sep=",",
                    header=None,
                    names=["CDR3_Sequence", "Cancer_Score"]
                )
                if score in file_data["Cancer_Score"].values:
                    files_with_score.append(filename)
            except:
                continue

        print(f"Score: {score:.6f}")
        print(f"Found in files: {files_with_score}\n")

    return outliers, lower_bound, upper_bound


def main(directory):
    save_directory = directory

    cancer_data = pd.DataFrame(columns=["CDR3_Sequence", "Cancer_Score", "CDR3_Length"])
    control_data = pd.DataFrame(columns=["CDR3_Sequence", "Cancer_Score", "CDR3_Length"])

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            data = pd.read_csv(
                os.path.join(directory, filename),
                sep=",",
                header=None,
                names=["CDR3_Sequence", "Cancer_Score"],
            )
            data["CDR3_Length"] = data["CDR3_Sequence"].apply(len)
            data["Source_File"] = filename
            data = data.dropna(axis=1, how="all")

            if "cancer" in filename:
                cancer_data = pd.concat([cancer_data, data], ignore_index=True)
            elif "control" in filename:
                control_data = pd.concat([control_data, data], ignore_index=True)

    counts = cancer_data["CDR3_Length"].value_counts()
    cancer_data_filtered = cancer_data[cancer_data["CDR3_Length"].isin(counts[counts >= 5].index)]

    counts_control = control_data["CDR3_Length"].value_counts()
    control_data_filtered = control_data[control_data["CDR3_Length"].isin(counts_control[counts_control >= 5].index)]

    cancer_data_filtered['Group'] = 'Cancer'
    control_data_filtered['Group'] = 'Control'
    combined_data = pd.concat([cancer_data_filtered, control_data_filtered])

    # Plot 1
    plt.figure(figsize=(20, 10))
    sns.boxplot(
        x="CDR3_Length",
        y="Cancer_Score",
        hue="Group",
        data=combined_data,
        showfliers=False
    )
    plt.title("Box Plots of Cancer Scores by CDR3 Sequence Length (Cancer vs. Control)")
    plt.xlabel("CDR3 Sequence Length")
    plt.ylabel("Cancer Score")
    plt.legend(title="Group")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(save_directory, "cancer_control_boxplots.png"))
    plt.close()

    # Plot 2
    csv_file_path = os.path.join(directory, "metric_scores.csv")
    data = pd.read_csv(csv_file_path)
    metrics = ["Mean Score"]

    plt.figure(figsize=(20, 10))
    plt.scatter(
        range(len(data["Mean Score"])),
        data["Mean Score"],
        alpha=0.6,
        c='blue'
    )
    plt.title("Distribution of Mean Scores Across Files")
    plt.xlabel("File Index")
    plt.ylabel("Mean Score")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_directory, "mean_scores_scatter.png"))
    plt.close()

    # Plot 3
    plt.figure(figsize=(24, 10))
    plt.boxplot([data[metric] for metric in metrics], labels=metrics)
    plt.title("Boxplots of Cancer CDR3 Scores")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "metric_boxplots.png"))
    plt.close()

    # Plot 4
    plt.figure(figsize=(20, 10))
    sns.histplot(
        combined_data["Cancer_Score"],
        bins=10,
        kde=True,
        color='purple',
        alpha=0.7
    )
    plt.title("Histogram of Individual Sequence Scores")
    plt.xlabel("Cancer Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "individual_sequence_scores_histogram.png"))
    plt.close()

    # Outlier analysis
    for metric in metrics:
        if metric != "Mean Score":
            continue
        outliers, _, _ = identify_outliers_with_files(data, metric, directory)
        outlier_file = os.path.join(save_directory, f"outliers_{metric.lower().replace(' ', '_')}.csv")
        outliers.to_csv(outlier_file, index=False)
        print(f"\nOutliers saved to: {outlier_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cancer/control scores and generate plots.")
    parser.add_argument(
        "--directory",
        type=str,
        default="/scratch/project/tcr_ml/gnn_release/model_2025_ccdi_only/PICA_scores/",
        help="Directory containing input .txt and .csv files. Defaults to preset path."
    )
    args = parser.parse_args()
    main(args.directory)
