import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Base directory for the project
base_directory = "/scratch/project/tcr_ml/GNN/model_training/trial_hyperparameter_optuna_zero"

# Directory where the files are stored
directory = f"{base_directory}/scores/"

# Directory to save the plots
save_directory = base_directory

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

# Plot box plots of Cancer Scores by CDR3 Length, grouped by Cancer and Control
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
plt.xticks(rotation=90)  # Rotate x-ticks for better readability

plt.show()
plt.savefig(save_directory + "/cancer_control_boxplots.png")

# Plot for CDR3 length distribution
plt.figure(figsize=(15, 10))
# Plot for cancer data
sns.histplot(
    cancer_data["CDR3_Length"], bins=30, color="r", stat="density", label="Cancer"
)
# Plot for control data
sns.histplot(
    control_data["CDR3_Length"], bins=30, color="b", stat="density", label="Control"
)
plt.xlim(5, 30)  # Adjust x-axis limits
plt.xticks(range(5, 30, 1))  # Set x-ticks to step by 1
plt.title("Distribution of CDR3 Sequence Lengths")
plt.xlabel("CDR3 Sequence Length")
plt.ylabel("Density")

plt.legend()  # Add a legend

plt.show()
plt.savefig(save_directory + "/length_distribution_normalized.png")
