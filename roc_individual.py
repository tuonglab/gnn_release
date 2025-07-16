import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(cancer_file, control_file):
    """
    Load cancer and control files and prepare them for ROC analysis.
    """
    try:
        cancer_df = pd.read_csv(cancer_file)
        control_df = pd.read_csv(control_file)
        
        if 'Mean Score' not in cancer_df.columns or 'Mean Score' not in control_df.columns:
            raise ValueError("One or both files missing 'Mean Score' column")
            
        cancer_scores = cancer_df['Mean Score'].values
        control_scores = control_df['Mean Score'].values
        
        cancer_labels = np.ones(len(cancer_scores))
        control_labels = np.zeros(len(control_scores))
        
        all_scores = np.concatenate([cancer_scores, control_scores])
        all_labels = np.concatenate([cancer_labels, control_labels])
        
        return all_scores, all_labels
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def plot_boxplot(cancer_scores, control_scores, output_dir, dataset):
    """
    Plot boxplot comparing cancer and control scores.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'boxplot_{dataset}.png')
    
    data_to_plot = [control_scores, cancer_scores]
    labels = ['Control', 'Cancer']
    
    plt.figure()
    plt.boxplot(data_to_plot, labels=labels, notch=True, patch_artist=True)
    plt.ylabel('Mean Score')
    plt.title('Mean Score Comparison: Cancer vs. Control')
    
    plt.savefig(output_path)
    plt.close()
    return output_path

def calculate_roc(scores, labels):
    """
    Calculate ROC curve and AUC score.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds

def plot_roc_curve(fpr, tpr, roc_auc, output_dir, dataset):
    """
    Plot ROC curve and save to specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'roc_{dataset}.png')
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.savefig(output_path)
    plt.close()
    return output_path

# File paths
control_file = '/scratch/project/tcr_ml/gnn_release/model_2025_sc_curated_control/val_control_scores/metric_scores.csv'
cancer_file = '/scratch/project/tcr_ml/gnn_release/model_2025_sc_curated_control/sarcoma_zero_scores/metric_scores.csv'

# Extract cancer and control dataset names and strip "_scores" suffix
cancer_folder = os.path.basename(os.path.dirname(cancer_file)).replace("_scores", "")
control_folder = os.path.basename(os.path.dirname(control_file)).replace("_scores", "")
dataset = f"{cancer_folder}_vs_{control_folder}"


try:
    scores, labels = load_and_prepare_data(cancer_file, control_file)
    fpr, tpr, roc_auc, thresholds = calculate_roc(scores, labels)

    # Split back to individual groups for boxplot
    control_df = pd.read_csv(control_file)
    cancer_df = pd.read_csv(cancer_file)
    
    control_scores = control_df['Mean Score'].values
    cancer_scores = cancer_df['Mean Score'].values

    print(f"\nROC Analysis Results:")
    print(f"AUC Score: {roc_auc:.4f}")
    
    # Determine model base directory and construct output path
    model_base_dir = os.path.abspath(os.path.join(cancer_file, "../../"))  # Go up two directories
    output_dir = os.path.join(model_base_dir, 'roc_comparisons')
    
    output_path = plot_roc_curve(fpr, tpr, roc_auc, output_dir, dataset)
    print(f"ROC curve plot saved as: {output_path}")

    # Plot boxplot
    output_path_box = plot_boxplot(cancer_scores, control_scores, output_dir, dataset)
    print(f"Boxplot saved as: {output_path_box}")
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"At optimal threshold:")
    print(f"True Positive Rate: {tpr[optimal_idx]:.4f}")
    print(f"False Positive Rate: {fpr[optimal_idx]:.4f}")
    
except Exception as e:
    print(f"Error: {str(e)}")
