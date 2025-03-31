import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


## ROC curve plot based on 
def load_and_prepare_data(cancer_file, control_file):
    """
    Load cancer and control files and prepare them for ROC analysis.
    """
    # Read the CSV files
    try:
        cancer_df = pd.read_csv(cancer_file)
        control_df = pd.read_csv(control_file)
        
        # Verify 'Mean Score' column exists
        if 'Mean Score' not in cancer_df.columns or 'Mean Score' not in control_df.columns:
            raise ValueError("One or both files missing 'Mean Score' column")
            
        # Extract scores
        cancer_scores = cancer_df['Mean Score'].values
        control_scores = control_df['Mean Score'].values
        
        # Create labels (1 for cancer, 0 for control)
        cancer_labels = np.ones(len(cancer_scores))
        control_labels = np.zeros(len(control_scores))
        
        # Combine scores and labels
        all_scores = np.concatenate([cancer_scores, control_scores])
        all_labels = np.concatenate([cancer_labels, control_labels])
        
        return all_scores, all_labels
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def calculate_roc(scores, labels):
    """
    Calculate ROC curve and AUC score.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve and save to file.
    """
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
    
    plt.savefig('roc_curve.png')
    plt.close()

# File paths - MODIFY THESE TO MATCH YOUR FILES
cancer_file = '/scratch/project/tcr_ml/gnn_release/model_2025_360/phs002517_scores/metric_scores.csv'  # Replace with your cancer file path
control_file = '/scratch/project/tcr_ml/gnn_release/model_2025_360/control_leftover_scores/metric_scores.csv'  # Replace with your control file path

try:
    # Load and prepare data
    scores, labels = load_and_prepare_data(cancer_file, control_file)
    
    # Calculate ROC
    fpr, tpr, roc_auc, thresholds = calculate_roc(scores, labels)
    
    # Print results
    print(f"\nROC Analysis Results:")
    print(f"AUC Score: {roc_auc:.4f}")
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc)
    print(f"ROC curve plot saved as: roc_curve.png")
        
    # Print optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"At optimal threshold:")
    print(f"True Positive Rate: {tpr[optimal_idx]:.4f}")
    print(f"False Positive Rate: {fpr[optimal_idx]:.4f}")
    
except Exception as e:
    print(f"Error: {str(e)}")