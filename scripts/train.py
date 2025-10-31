from pathlib import Path

from tcrgnn.utils.device import get_device, set_seed

from tcrgnn import GATv2, TrainConfig, TrainPaths, train
from tcrgnn.utils.data_loading import load_train_data


def run():
    device = get_device()
    cfg = TrainConfig(epochs=500, batch_size=256, patience=15, seed=111)
    paths = TrainPaths(
        model_dir=Path("model_2025_boltz_111"), best_name="best_model.pt"
    )
    set_seed(cfg.seed)

    cancer_dirs = [
        "/scratch/project/tcr_ml/gnn_release/dataset_boltz/blood_tissue_predictions/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_boltz/ccdi_boltz/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_boltz/scTRB_predictions/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_boltz/tumor_tissue_predictions/processed",
    ]
    control_dirs = [
        "/scratch/project/tcr_ml/gnn_release/dataset_boltz/curated/processed",
    ]

    samples = load_train_data(cancer_dirs, control_dirs)
    nfeat = samples[0][0].num_node_features
    model = GATv2(nfeat=nfeat, nhid=375, nclass=2, dropout=0.17).to(device)
    train(model, samples, cfg, paths, device)


if __name__ == "__main__":
    run()
