import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Base directory for the project
base_directory = "/scratch/project/tcr_ml/gnn_release/model"

# Directory where the files are stored
directory = "/scratch/project/tcr_ml/gnn_release/legacy_model/scores/"

# Directory to save the plots
save_directory = directory

def identify_outliers_with_files(data, column, directory):
    # Calculate Q1, Q3, and IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = data[
        (data[column] < lower_bound) | 
        (data[column] > upper_bound)
    ]
    
    print(f"\nOutliers for {column}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Lower bound: {lower_bound:.4f}")
    print(f"Upper bound: {upper_bound:.4f}")
    
    # Get list of all relevant files in directory
    all_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    # For each outlier value
    print("\nOutlier values and their source files:")
    for score in outliers[column].sort_values(ascending=False):
        files_with_score = []
        # Check each file
        for filename in all_files:
            file_path = os.path.join(directory, filename)
            try:
                file_data = pd.read_csv(
                    file_path,
                    sep=",",
                    header=None,
                    names=["CDR3_Sequence", "Cancer_Score"]
                )
                # Check if this score exists in the file
                if score in file_data["Cancer_Score"].values:
                    files_with_score.append(filename)
            except:
                continue
                
        print(f"Score: {score:.6f}")
        print(f"Found in files: {files_with_score}\n")

    return outliers, lower_bound, upper_bound

# Initialize empty dataframes for cancer and control scores
cancer_data = pd.DataFrame(columns=["CDR3_Sequence", "Cancer_Score", "CDR3_Length"])
control_data = pd.DataFrame(columns=["CDR3_Sequence", "Cancer_Score", "CDR3_Length"])

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # Load the data
        data = pd.read_csv(
            directory + filename,
            sep=",",
            header=None,
            names=["CDR3_Sequence", "Cancer_Score"],
        )
        # Calculate the length of each CDR3 sequence
        data["CDR3_Length"] = data["CDR3_Sequence"].apply(len)

        # Drop empty or all-NA columns
        data = data.dropna(axis=1, how="all")

        # Append to the appropriate dataframe
        if "cancer" in filename:
            cancer_data = pd.concat([cancer_data, data], ignore_index=True)
        elif "control" in filename:
            control_data = pd.concat([control_data, data], ignore_index=True)

# Filter out CDR3 lengths with less than 5 sequences
counts = cancer_data["CDR3_Length"].value_counts()
cancer_data_filtered = cancer_data[
    cancer_data["CDR3_Length"].isin(counts[counts >= 5].index)
]

counts_control = control_data["CDR3_Length"].value_counts()
control_data_filtered = control_data[
    control_data["CDR3_Length"].isin(counts_control[counts_control >= 5].index)
]

# Combine cancer and control data into a single dataframe for plotting
cancer_data_filtered['Group'] = 'Cancer'
control_data_filtered['Group'] = 'Control'
combined_data = pd.concat([cancer_data_filtered, control_data_filtered])

# Plot 1: Box plots of Cancer Scores by CDR3 Length
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
plt.savefig(save_directory + "/cancer_control_boxplots.png")
plt.close()


# Load metric scores data
csv_file_path = f'{directory}/metric_scores.csv'
data = pd.read_csv(csv_file_path)

# Select metrics columns for plotting
metrics = [ "Mean Score"]

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
plt.savefig(save_directory + "/mean_scores_scatter.png")
plt.close()

# Plot 4: Create boxplot with highlighted outliers
plt.figure(figsize=(24, 10))
bp = plt.boxplot([data[metric] for metric in metrics], labels=metrics)
plt.title("Boxplots of Cancer CDR3 Scores by Metric with Outliers")
plt.ylabel("Score")
plt.xticks(rotation=45)

# Highlight outliers in red
for i in range(len(metrics)):
    outliers = bp['fliers'][i]
    outliers.set_color('red')
    
plt.tight_layout()
plt.savefig(f"{save_directory}/metric_boxplots_with_outliers.png")
plt.close()

# Perform outlier analysis for each metric
for metric in metrics:
    if metric!="Mean Score":
        continue
    outliers, _, _ = identify_outliers_with_files(data, metric, directory)
    
    # Save outliers to CSV
    outlier_file = f"{save_directory}/outliers_{metric.lower().replace(' ', '_')}.csv"
    outliers.to_csv(outlier_file, index=False)
    print(f"\nOutliers saved to: {outlier_file}")