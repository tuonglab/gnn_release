import os
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from scipy.stats import zscore
from uncertainty_evaluation.train_uncertainity import GATv2Heteroscedastic, device
from graph_generation.graph import load_graphs
from pathlib import Path




MODEL_FILE = "/scratch/project/tcr_ml/gnn_release/uncertainty_evaluation/model_2025_uncertainty_curated/best_model.pt"
@torch.no_grad()
def mc_dropout_predict(model, data, T=20):
    model.train()  # force dropout on
    probs_list, log_vars_list = [],[]

    for _ in range(T):
        logits, log_var = model(data.x, data.edge_index, data.batch)
        probs = softmax(logits, dim=1)
        probs_list.append(probs)
        log_vars_list.append(log_var)

    probs_stack = torch.stack(probs_list)  # [T, B, C]
    log_vars_stack = torch.stack(log_vars_list)  # [T, B, 1]

    mean_probs = probs_stack.mean(dim=0)
    mean_log_vars = log_vars_stack.mean(dim=0).squeeze(-1)

    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
    expected_entropy = -torch.mean(torch.sum(probs_stack * torch.log(probs_stack + 1e-8), dim=2), dim=0)
    mi = entropy - expected_entropy

    return mean_probs, mean_log_vars, entropy, mi


def evaluate_mc_dropout(model, dataset, T=20, return_individual=False):
    model.eval()
    results = []
    per_graph_records = []

    for sample_idx, sample_graphs in enumerate(dataset):
        entropy_list, mi_list, var_list = [], [], []
        for graph_idx, g in enumerate(sample_graphs):
            g = g.to(device)
            probs, log_var, entropy, mi = mc_dropout_predict(model, g, T=T)
            var = torch.exp(log_var)
            probs = probs

            entropy_list.extend(entropy.tolist())
            mi_list.extend(mi.tolist())
            var_list.extend(var.tolist())

            if return_individual:
                for i in range(len(entropy)):
                    per_graph_records.append({
                        "sample_id": sample_idx,
                        "graph_idx": graph_idx,
                        "subgraph_idx": i,
                        "entropy": entropy[i].item(),
                        "mutual_info": mi[i].item(),
                        "aleatoric_var": var[i].item(),
                        "pred_class": torch.argmax(probs[i]).item(),
                        "pred_prob_0": probs[i][0].item(),
                        "pred_prob_1": probs[i][1].item()
                    })

        results.append({
            "sample_id": sample_idx,
            "mean_entropy": np.mean(entropy_list),
            "mean_mutual_info": np.mean(mi_list),
            "mean_aleatoric_var": np.mean(var_list),
            "num_graphs": len(sample_graphs)
        })

    df = pd.DataFrame(results)
    df["z_entropy"] = zscore(df["mean_entropy"])
    df["z_mutual_info"] = zscore(df["mean_mutual_info"])
    df["z_aleatoric_var"] = zscore(df["mean_aleatoric_var"])

    if return_individual:
        return df, pd.DataFrame(per_graph_records)
    else:
        return df


def print_uncertainty_summary(df, print_mode="all", show_all_rows=False):
    print("\n=== Uncertainty Summary ===")

    if print_mode == "epistemic":
        columns = ["sample_id", "mean_mutual_info"]
    elif print_mode == "aleatoric":
        columns = ["sample_id", "mean_aleatoric_var"]
    else:
        columns = ["sample_id", "mean_entropy", "mean_mutual_info", "mean_aleatoric_var"]

    if show_all_rows:
        pd.set_option("display.max_rows", None)
        print(df[columns])
        pd.reset_option("display.max_rows")
    else:
        print(df[columns].describe())


def load_eval_data(path):
    dataset = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            graphs = load_graphs(file_path)
            dataset.append(graphs)
    return dataset


def main():


    test_dir = "/scratch/project/tcr_ml/gnn_release/test_data_v2/val_control/processed"
    prefix = "/scratch/project/tcr_ml/gnn_release"

    rel_parts = Path(test_dir).relative_to(prefix).parts
    dataset_name = rel_parts[1]  # index 1 is 'd360'



    test_set = load_eval_data(test_dir)

    model = GATv2Heteroscedastic(
        nfeat=test_set[0][0].num_node_features,
        nhid=375,
        nclass=2
    ).to(device)

    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    results_df, per_graph_df = evaluate_mc_dropout(model, test_set, T=25, return_individual=True)

    # Summary statistics
    print_uncertainty_summary(results_df, print_mode="all")
    print_uncertainty_summary(results_df, print_mode="all", show_all_rows=True)

    # Define CSV output directory inside model directory
    model_dir = os.path.dirname(MODEL_FILE)
    csv_dir = os.path.join(model_dir, "uncertainty_csv")
    graph_csv_dir = os.path.join(model_dir, "graph_level_uncertainty")
    os.makedirs(graph_csv_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Save per-sample summary
    output_filename = f"uncertainty_results_{dataset_name}.csv"
    output_path = os.path.join(csv_dir, output_filename)
    results_df.to_csv(output_path, index=False)

    # Save per-graph metrics
    graph_output_filename = f"uncertainty_graph_level_{dataset_name}.csv"
    graph_output_path = os.path.join(graph_csv_dir, graph_output_filename)
    per_graph_df.to_csv(graph_output_path, index=False)

    print(f"\n✅ Per-sample summary exported to: {output_path}")
    print(f"✅ Per-graph scores exported to: {graph_output_path}")


if __name__ == "__main__":
    main()
