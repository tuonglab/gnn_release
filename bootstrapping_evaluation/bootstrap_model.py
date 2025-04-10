import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# --------------------------------------------------
# Basic Setup
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(46)
np.random.seed(46)

MODEL_PATH = "model_2025_isacs_only"
MODEL_NAME = "best_model.pt"
MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)
os.makedirs(MODEL_PATH, exist_ok=True)

# --------------------------------------------------
# GATv2 Definition
# --------------------------------------------------
class GATv2(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, temperature=1.0):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(nfeat, nhid, heads=16, dropout=dropout, concat=False)
        self.conv2 = GATv2Conv(nhid, nhid, heads=16, dropout=dropout, concat=False)
        self.classifier = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.temperature = temperature

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x /= self.temperature
        return x

# --------------------------------------------------
# Data Loading Function
# --------------------------------------------------
def load_graphs(file_path):
    """
    Stub for 'load_graphs' function.
    Replace with your real implementation that
    loads and returns a list of PyG 'Data' objects.
    """
    # Placeholder: must return List[Data]
    raise NotImplementedError("Replace this with your real 'load_graphs' definition.")

def load_train_data(cancer_paths: list, control_paths: list):
    training_set = []
    for cancer_path in cancer_paths:
        if os.path.isdir(cancer_path):
            for filename in os.listdir(cancer_path):
                file_path = os.path.join(cancer_path, filename)
                graphs = load_graphs(file_path)
                training_set.append(graphs)

    for control_path in control_paths:
        if os.path.isdir(control_path):
            for filename in os.listdir(control_path):
                file_path = os.path.join(control_path, filename)
                graphs = load_graphs(file_path)
                training_set.append(graphs)

    # 'training_set' is a list of lists; flatten if needed:
    # Possibly each 'load_graphs' returns a list of Data objects. 
    # So if you want a single list of Data objects, do:
    flat_dataset = []
    for item in training_set:
        flat_dataset.extend(item)
    return flat_dataset

# --------------------------------------------------
# Brier Score Helper
# --------------------------------------------------
def brier_score(probs, targets):
    return torch.mean((probs - targets) ** 2).item()

# --------------------------------------------------
# Single-Run Training
# --------------------------------------------------
def train_single_run(dataset, num_epochs=100, patience=15, lr=0.0005, weight_decay=0.25, hidden_dim=375, dropout=0.17):
    """
    Trains one GATv2 model on the entire 'dataset'.
    Returns (model, best_loss, None).
    """
    nfeat = dataset[0].num_node_features  # dataset is a list of PyG 'Data'
    model = GATv2(nfeat=nfeat, nhid=hidden_dim, nclass=2, dropout=dropout).to(device)

    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for data in data_loader:
            optimizer.zero_grad()

            # data here is a single batch of multiple graphs
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
            # out shape: (batch_size, 2)
            out_pos = out[:, 1]
            
            loss = criterion(out_pos, data.y.float().to(device))
            loss.backward()
            optimizer.step()

            batch_size = data.y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, best_loss, None

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------
def evaluate_model(model, dataset):
    """
    Evaluates 'model' on the entire 'dataset'.
    Returns (avg_BCE, avg_Brier).
    """
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    total_loss = 0.0
    total_brier = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in data_loader:
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
            out_pos = out[:, 1]

            probs = torch.sigmoid(out_pos)
            labels = data.y.float().to(device)

            loss = criterion(out_pos, labels)
            brier = brier_score(probs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_brier += brier * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_brier = total_brier / total_samples
    return avg_loss, avg_brier

# --------------------------------------------------
# 0.632 Bootstrap
# --------------------------------------------------
def bootstrap_632(dataset, B=50, num_epochs=100, patience=15):
    """
    Returns mean .632 estimates (BCE Loss and Brier)
    over B bootstrap iterations.
    """
    N = len(dataset)

    # Apparent error: train on entire dataset, evaluate on entire dataset
    full_model, _, _ = train_single_run(dataset, num_epochs=num_epochs, patience=patience)
    e_app_loss, e_app_brier = evaluate_model(full_model, dataset)

    e_632_losses = []
    e_632_briers = []

    for _ in range(B):
        bootstrap_indices = np.random.choice(range(N), size=N, replace=True)
        in_bag_data = [dataset[i] for i in bootstrap_indices]
        oob_indices = set(range(N)) - set(bootstrap_indices)
        
        if len(oob_indices) == 0:
            # fallback if no OOB data
            oob_data = dataset
        else:
            oob_data = [dataset[i] for i in oob_indices]

        model_b, _, _ = train_single_run(in_bag_data, num_epochs=num_epochs, patience=patience)
        e_oob_loss, e_oob_brier = evaluate_model(model_b, oob_data)

        # .632 weighting
        loss_632 = 0.368 * e_app_loss + 0.632 * e_oob_loss
        brier_632 = 0.368 * e_app_brier + 0.632 * e_oob_brier

        e_632_losses.append(loss_632)
        e_632_briers.append(brier_632)

    mean_632_loss = np.mean(e_632_losses)
    mean_632_brier = np.mean(e_632_briers)
    return mean_632_loss, mean_632_brier

# --------------------------------------------------
# 0.632+ Bootstrap
# --------------------------------------------------
def compute_noinfo_baselines(dataset):
    """
    Computes the "no-information" references for:
    - Brier score
    - BCE loss
    by using the fraction of positives in the entire dataset.
    """
    all_labels = []
    for data in dataset:
        # data.y is a tensor of 0/1
        all_labels.extend(data.y.tolist())

    all_labels = np.array(all_labels)
    p = np.mean(all_labels)

    # Clip p in case it's 0 or 1 for BCE
    eps = 1e-8
    p_clamped = np.clip(p, eps, 1 - eps)

    # No-info Brier for binary classification: p(1-p)
    noinf_brier = p * (1.0 - p)
    
    # No-info BCE = -[ p ln(p) + (1-p) ln(1-p) ]
    noinf_bce = -(p_clamped * np.log(p_clamped) + (1 - p_clamped) * np.log(1 - p_clamped))

    return noinf_brier, noinf_bce

def _632plus_formula(e_app, e_oob, e_noinf):
    """
    Implements the .632+ formula for one metric (loss or Brier).
    
      R = (e_oob - e_app) / (e_noinf - e_app), if e_noinf != e_app
      w = 0.632 / (1 - 0.368 * R)
      
      if R < 0: e_632+ = 0.368 e_app + 0.632 e_oob
      elif R > 1: e_632+ = e_oob
      else: e_632+ = e_app + w * (e_oob - e_app)
    """
    # Avoid divide-by-zero if e_noinf == e_app:
    if abs(e_noinf - e_app) < 1e-12:
        # fallback to .632
        return 0.368 * e_app + 0.632 * e_oob

    R = (e_oob - e_app) / (e_noinf - e_app)

    if R < 0:
        # fallback to .632
        return 0.368 * e_app + 0.632 * e_oob
    if R > 1:
        return e_oob

    w = 0.632 / (1.0 - 0.368 * R)
    return e_app + w * (e_oob - e_app)

def bootstrap_632_plus(dataset, B=50, num_epochs=100, patience=15):
    """
    Returns mean .632+ estimates for both BCE loss and Brier score
    over B bootstrap iterations.
    """
    N = len(dataset)

    # 1) Apparent error on entire dataset
    full_model, _, _ = train_single_run(dataset, num_epochs=num_epochs, patience=patience)
    e_app_loss, e_app_brier = evaluate_model(full_model, dataset)

    # 2) No-information references
    e_noinf_brier, e_noinf_loss = compute_noinfo_baselines(dataset)
    # NOTE: naming: e_noinf_loss corresponds to BCE, e_noinf_brier is Brier.

    e_632plus_losses = []
    e_632plus_briers = []

    for _ in range(B):
        # bootstrap sample
        bootstrap_indices = np.random.choice(range(N), size=N, replace=True)
        in_bag_data = [dataset[i] for i in bootstrap_indices]
        oob_indices = set(range(N)) - set(bootstrap_indices)
        
        if len(oob_indices) == 0:
            oob_data = dataset
        else:
            oob_data = [dataset[i] for i in oob_indices]

        # train on in-bag
        model_b, _, _ = train_single_run(in_bag_data, num_epochs=num_epochs, patience=patience)
        # evaluate on OOB
        e_oob_loss, e_oob_brier = evaluate_model(model_b, oob_data)

        # .632+ for BCE loss
        e_632plus_loss = _632plus_formula(e_app_loss, e_oob_loss, e_noinf_loss)
        # .632+ for Brier
        e_632plus_brier = _632plus_formula(e_app_brier, e_oob_brier, e_noinf_brier)

        e_632plus_losses.append(e_632plus_loss)
        e_632plus_briers.append(e_632plus_brier)

    mean_632plus_loss = np.mean(e_632plus_losses)
    mean_632plus_brier = np.mean(e_632plus_briers)
    return mean_632plus_loss, mean_632plus_brier

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # Example file paths (replace with real ones):
    train_cancer_directories = [
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/blood_tissue/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/scTRB/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/tumor_tissue/processed",
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/ccdi/processed",
    ]
    train_control_directories = [
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/control/processed"
    ]

    # Load training data
    train_set = load_train_data(train_cancer_directories, train_control_directories)
    print(f"Loaded dataset with {len(train_set)} total graphs.")

    B = 20
    num_epochs = 200
    patience = 15

    # ---- .632 original ----
    mean_632_loss, mean_632_brier = bootstrap_632(
        dataset=train_set, B=B, num_epochs=num_epochs, patience=patience
    )
    print(f"\n=== .632 Bootstrap ===")
    print(f"Mean .632 Loss over {B} bootstraps:  {mean_632_loss:.4f}")
    print(f"Mean .632 Brier Score over {B} bootstraps: {mean_632_brier:.4f}")

    # ---- .632+ improved ----
    mean_632plus_loss, mean_632plus_brier = bootstrap_632_plus(
        dataset=train_set, B=B, num_epochs=num_epochs, patience=patience
    )
    print(f"\n=== .632+ Bootstrap ===")
    print(f"Mean .632+ Loss over {B} bootstraps:  {mean_632plus_loss:.4f}")
    print(f"Mean .632+ Brier Score over {B} bootstraps: {mean_632plus_brier:.4f}")


if __name__ == "__main__":
    main()
