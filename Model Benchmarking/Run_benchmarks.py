# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Setting up comparative benchmarking suites across diverse deep learning architectures (CNN 1D, CNN-LSTM, TCN, Transformers, and non-shapelet GNNs).
- Executing repeated training runs to collect latency, peak memory, parameter count, FLOPs, and accuracy statistics.
- Generating LaTeX summary tables and exporting statistical comparison files for review.

Note: Any reuse of this code should be authorized by the code author.
"""

import os
import sys


import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader as SeqDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Resolve package directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Package Imports
from Dataloader import load_raw_data, get_sequence_datasets, get_graph_datasets, extract_all_features
from Network import (CNN1D, CNNLSTM, TCN, TransformerClassifier,
                              GATWithoutShapelets, GCNWithoutShapelets, GNNWithAttention)
from Visualization import (generate_plots, plot_model_confusion_matrices, 
                                    plot_training_val_curves, plot_cumulative_val_curves,
                                    plot_model_complexities, generate_resource_comparison_plots)
from Utils import (get_peak_memory, reset_memory_tracker, get_parameter_count,
                            estimate_flops, train_and_eval_dl, compute_95_ci,
                            run_statistical_test, generate_latex_table)


# =============================================================================
# 1. Benchmarking Model Toggles (True = Enable, False = Disable)
# =============================================================================
RUN_SHAPELET_GAT = False
RUN_CNN_1D = True
RUN_CNN_LSTM = True
RUN_TCN = True
RUN_TRANSFORMER = True
RUN_GAT_NO_SHAPELETS = True
RUN_GCN_NO_SHAPELETS = True
RUN_RANDOM_FOREST = False


# =============================================================================
# 2. Dataset Setup & Training Hyperparameters (Easily Customizable)
# =============================================================================
SEED = 42               # Global base seed for dataset generation/loading/balancing
SEEDS = 1               # Number of random seeds to execute (for CIs and stats)
EPOCHS = 300           # Number of training epochs for PyTorch Deep Learning models
BATCH_SIZE = 256        # Batch size for deep learning DataLoader
LEARNING_RATE = 0.005   # Learning rate for Adam optimizer
TEST_SIZE = 0.20        # Proportion of dataset reserved for final testing (e.g. 0.20 = 20%)
VAL_SIZE = 0.15         # Proportion of dataset reserved for validation (e.g. 0.15 = 15%)


# =============================================================================
# 3. Model Architecture Details
# =============================================================================
SHAPELET_LEN = 50       # Length of learned temporal subsequences
NUM_SHAPELETS = 10      # Number of shapelets to learn per channel
HIDDEN_CHANNELS = 16    # Hidden channels dimension in GAT/GCN layers


# =============================================================================
# 4. Reporting & Visualization Toggles (True / False)
# =============================================================================
GENERATE_COMPARISON_PLOTS = True   # Side-by-side performance bar plots with 95% CI
GENERATE_LATEX_TABLE = True        # Manuscript-ready LaTeX table compilation
GENERATE_CONFUSION_MATRICES = True # Absolute & normalized confusion matrices


# Centralized Figures Output directory
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "Figures")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =============================================================================
# Main Benchmarking Loop
# =============================================================================

def print_pretty_complexity_table(df_agg, comp_dir):
    """
    Description:
        Generates a professionally formatted, double-ruled Unicode ASCII table comparing model parameters, file footprint sizes, and relative complexity classes. Prints it to stdout and saves it as a text file.
    Purpose:
        To present structured model resource profiles to researchers and reviewers clearly.
    Input Types:
        - df_agg (pandas.DataFrame): Aggregated performance and resource metrics of the benchmarked models.
        - comp_dir (str): Folder path where the text table file is saved.
    Output Types:
        - None: Directly writes to terminal and disk.
    """
    lines = []
    lines.append("╔" + "═" * 24 + "╦" + "═" * 21 + "╦" + "═" * 22 + "╦" + "═" * 23 + "╗")
    lines.append("║ " + f"{'Model Name':<22} ║ {'Trainable Params':<19} ║ {'Footprint (KB)':<20} ║ {'Complexity Class':<21} ║")
    lines.append("╠" + "═" * 24 + "╬" + "═" * 21 + "╬" + "═" * 22 + "╬" + "═" * 23 + "╣")
    
    for idx, row in df_agg.iterrows():
        model_name = row["Model"]
        params = row["Parameters"]
        
        if params == "N/A":
            params_str = "N/A"
            kb_str = "N/A"
            complexity = "Classical ML (Non-DL)"
        else:
            p_val = int(params)
            params_str = f"{p_val:,}"
            kb_val = (p_val * 4) / 1024.0 # FP32 model size
            kb_str = f"{kb_val:.2f} KB"
            if p_val < 10000:
                complexity = "Ultra-Lightweight"
            elif p_val < 50000:
                complexity = "Lightweight"
            elif p_val < 100000:
                complexity = "Medium"
            else:
                complexity = "Heavyweight"
                
        lines.append("║ " + f"{model_name:<22} ║ {params_str:>19} ║ {kb_str:>20} ║ {complexity:<21} ║")
        
    lines.append("╚" + "═" * 24 + "╩" + "═" * 21 + "╩" + "═" * 22 + "╩" + "═" * 23 + "╝")
    
    table_text = "\n".join(lines)
    print("\n" + "=" * 94)
    print("📈 PRETTY MODEL COMPLEXITY & MEMORY FOOTPRINT TABLE 📈")
    print("=" * 94)
    print(table_text)
    print("=" * 94 + "\n")
    
    with open(os.path.join(comp_dir, "master_complexity_table.txt"), "w", encoding="utf-8") as f:
        f.write(table_text)

def create_detailed_comparison_csv(df_agg, toggles, comp_dir):
    """
    Description:
        Creates a detailed comma-separated value (CSV) log mapping models to their parameter sizes, loss functions, input structures, validation training speeds, and estimated FLOP metrics.
    Purpose:
        To store raw profiling data for systematic visualization and external plotting.
    Input Types:
        - df_agg (pandas.DataFrame): Aggregated performance and complexity metrics.
        - toggles (dict): Model enable/disable boolean flags dictionary.
        - comp_dir (str): Folder path where the final CSV file is saved.
    Output Types:
        - None: Writes the CSV file to disk.
    """
    comparison_rows = []
    
    # Input dimension mapping
    input_dims = {
        "Shapelet-GAT": "19 nodes x [2, 500] (raw) -> 19 nodes x 20 (shapelet)",
        "1D-CNN": "[2, 5000]",
        "CNN-LSTM": "[2, 5000]",
        "TCN": "[2, 5000]",
        "Transformer": "[2, 5000]",
        "GAT-No-Shapelets": "19 nodes x [2, 500]",
        "GCN-No-Shapelets": "19 nodes x [2, 500]",
        "Random-Forest": "[44] (extracted features)"
    }
    
    # Loss function mapping
    loss_functions = {
        "Shapelet-GAT": "CrossEntropyLoss",
        "1D-CNN": "CrossEntropyLoss",
        "CNN-LSTM": "CrossEntropyLoss",
        "TCN": "CrossEntropyLoss",
        "Transformer": "CrossEntropyLoss",
        "GAT-No-Shapelets": "CrossEntropyLoss",
        "GCN-No-Shapelets": "CrossEntropyLoss",
        "Random-Forest": "Gini Impurity (Trees)"
    }
    
    for idx, row in df_agg.iterrows():
        model_name = row["Model"]
        if not toggles.get(model_name, False):
            continue
            
        params = row["Parameters"]
        params_str = f"{int(params):,}" if params != "N/A" else "N/A"
        
        flops = row["FLOPs"]
        flops_str = f"{int(flops):,}" if flops != "N/A" else "N/A"
        
        acc_str = f"{row['Accuracy_mean']*100:.2f}% ± {row['Accuracy_ci']*100:.2f}%"
        train_str = f"{row['Train_Time_mean']:.2f} ± {row['Train_Time_ci']:.2f} s"
        infer_str = f"{row['Infer_Time_mean']:.3f} ± {row['Infer_Time_ci']:.3f} ms"
        
        comparison_rows.append({
            "Model": model_name,
            "Parameters": params_str,
            "Accuracy": acc_str,
            "Loss Function": loss_functions.get(model_name, "N/A"),
            "Input Dimension": input_dims.get(model_name, "N/A"),
            "Training Time": train_str,
            "Inference Time": infer_str,
            "FLOPs": flops_str
        })
        
    df_comp = pd.DataFrame(comparison_rows)
    save_path = os.path.join(comp_dir, "model_complexity_performance_comparison.csv")
    df_comp.to_csv(save_path, index=False)
    print(f"[REPORT] Created detailed complexity CSV at Figures/Comparison/model_complexity_performance_comparison.csv")


def main():
    """
    Description:
        The main execution function driving the reviewer benchmarking pipeline. Parses command line arguments, handles multi-seed sequence and graph splits, runs cross-model training, and generates overall comparative figures.
    Purpose:
        To run standard comparative benchmarking experiments deterministically to evaluate the performance improvement of the proposed Shapelet-GAT.
    Input Types:
        - None: Parses CLI arguments from `sys.argv` internally.
    Output Types:
        - None: Saves figures, logs, matrices, and LaTeX tables directly.
    """
    parser = argparse.ArgumentParser(description="Reviewer Benchmarking Pipeline")
    parser.add_argument("--debug", action="store_true", help="Run a quick debug dry-run (1 epoch, 200 samples)")
    args = parser.parse_args()
    
    # Read boolean flags from local config
    toggles = {
        "Shapelet-GAT": RUN_SHAPELET_GAT,
        "1D-CNN": RUN_CNN_1D,
        "CNN-LSTM": RUN_CNN_LSTM,
        "TCN": RUN_TCN,
        "Transformer": RUN_TRANSFORMER,
        "GAT-No-Shapelets": RUN_GAT_NO_SHAPELETS,
        "GCN-No-Shapelets": RUN_GCN_NO_SHAPELETS,
        "Random-Forest": RUN_RANDOM_FOREST
    }
    
    # Debug mode overrides
    epochs = 1 if args.debug else EPOCHS
    seeds_count = 2 if args.debug else SEEDS
    batch_size = 128 if args.debug else BATCH_SIZE
    
    print("=" * 60)
    print("🚀 REVIEWER BENCHMARKING PIPELINE CONFIGURATION 🚀")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Base Seed: {SEED} | Seeds: {seeds_count} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {LEARNING_RATE}")
    print(f"Splits Config: Test split = {TEST_SIZE*100:.1f}% | Validation split = {VAL_SIZE*100:.1f}% | Train split = {(1.0-TEST_SIZE-VAL_SIZE)*100:.1f}%")
    print(f"Proposed GNN Configs: Shapelet Len = {SHAPELET_LEN} | Num Shapelets = {NUM_SHAPELETS} | GAT/GCN Hidden Channels = {HIDDEN_CHANNELS}")
    print("=" * 60)
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    D1, D2, Y, class_labels, num_classes, label_encoder = load_raw_data(seed=SEED)
    if args.debug:
        print("[DEBUG] Slicing dataset to 200 samples for validation...")
        D1, D2, Y = D1[:200], D2[:200], Y[:200]
        
    seeds_list = [SEED + i for i in range(seeds_count)]
    seed_results = []
    last_seed_preds = {}
    all_seed_preds = {m: [] for m, active in toggles.items() if active}
    
    # Structure to track epoch histories across seeds for learning dynamics plots
    model_histories = {m: {"losses": [], "stds": [], "val_accs": [], "val_f1s": [], "val_rocs": []} for m, active in toggles.items() if active}
    
    # Core Main Model-by-Model and Seed-by-Seed Loop
    models_run = [m for m, active in toggles.items() if active]
    
    for model_name in models_run:
        print("\n" + "=" * 80)
        print(f"🚀 BENCHMARKING MODEL: {model_name} (Across {seeds_count} Seeds)")
        print("=" * 80)
        
        # Structure model-specific directory
        model_dir = os.path.join(OUTPUT_FOLDER, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        for s_idx, seed in enumerate(seeds_list):
            print(f"\n🌱 [SEED {s_idx+1}/{seeds_count}] Random Seed = {seed}...")
            
            # Prepare datasets with validation split option
            if VAL_SIZE > 0:
                # First split into train_val and test
                X_train_val_seq, X_test_seq, y_train_val_seq, y_test_seq = get_sequence_datasets(D1, D2, Y, test_split=TEST_SIZE, seed=seed)
                # Stratifiably partition train_val into train and val
                val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
                from sklearn.model_selection import train_test_split as skl_split
                X_train_seq, X_val_seq, y_train_seq, y_val_seq = skl_split(
                    X_train_val_seq, y_train_val_seq, test_size=val_ratio, stratify=y_train_val_seq, random_state=seed
                )
                
                # Graphs split
                train_val_graphs, test_graphs = get_graph_datasets(D1, D2, Y, test_split=TEST_SIZE, seed=seed)
                train_val_labels = [g.y.item() for g in train_val_graphs]
                train_graphs, val_graphs = skl_split(
                    train_val_graphs, test_size=val_ratio, stratify=train_val_labels, random_state=seed
                )
            else:
                X_train_seq, X_test_seq, y_train_seq, y_test_seq = get_sequence_datasets(D1, D2, Y, test_split=TEST_SIZE, seed=seed)
                X_val_seq, y_val_seq = X_test_seq, y_test_seq
                
                train_graphs, test_graphs = get_graph_datasets(D1, D2, Y, test_split=TEST_SIZE, seed=seed)
                val_graphs = test_graphs
            
            # DL Sequence Data Loaders
            train_seq_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.long))
            val_seq_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.long))
            test_seq_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.long))
            
            train_loader_seq = SeqDataLoader(train_seq_dataset, batch_size=batch_size, shuffle=True)
            val_loader_seq = SeqDataLoader(val_seq_dataset, batch_size=batch_size, shuffle=False)
            test_loader_seq = SeqDataLoader(test_seq_dataset, batch_size=batch_size, shuffle=False)
            
            # Graph Data Loaders
            train_loader_graph = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
            val_loader_graph = PyGDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
            test_loader_graph = PyGDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
            
            # Execute training based on model name
            if model_name == "Shapelet-GAT":
                print("--- Training Proposed Shapelet-GAT Model ---")
                model = GNNWithAttention(
                    in_channels=NUM_SHAPELETS * 2,
                    hidden_channels=HIDDEN_CHANNELS,
                    out_channels=num_classes,
                    shapelet_len=SHAPELET_LEN,
                    num_shapelets=NUM_SHAPELETS
                ).to(device)
                param_count = get_parameter_count(model)
                flops = estimate_flops(model, "graph", device)
                metrics, train_time, infer_time, mem = train_and_eval_dl(model, "graph", train_loader_graph, val_loader_graph, test_loader_graph, device, epochs, lr=LEARNING_RATE)
                print(f"  F1: {metrics['F1']:.4f} | Train: {train_time:.2f}s | Params: {param_count:,}")
                seed_results.append({"Seed": seed, "Model": "Shapelet-GAT", **metrics, "Train_Time": train_time, "Infer_Time": infer_time, "Memory_MB": mem, "Parameters": param_count, "FLOPs": flops})
                last_seed_preds["Shapelet-GAT"] = (metrics["preds"], metrics["labels"])
                all_seed_preds["Shapelet-GAT"].append((metrics["preds"], metrics["labels"]))
                model_histories["Shapelet-GAT"]["losses"].append(metrics["epoch_losses"])
                model_histories["Shapelet-GAT"]["stds"].append(metrics["epoch_stds"])
                model_histories["Shapelet-GAT"]["val_accs"].append(metrics["epoch_val_accs"])
                model_histories["Shapelet-GAT"]["val_f1s"].append(metrics["epoch_val_f1s"])
                model_histories["Shapelet-GAT"]["val_rocs"].append(metrics["epoch_val_rocs"])
                del model; gc.collect(); torch.cuda.empty_cache()
                
            elif model_name == "1D-CNN":
                print("--- Training 1D-CNN Baseline ---")
                model = CNN1D(in_channels=2, num_classes=num_classes).to(device)
                param_count = get_parameter_count(model)
                flops = estimate_flops(model, "sequence", device)
                metrics, train_time, infer_time, mem = train_and_eval_dl(model, "sequence", train_loader_seq, val_loader_seq, test_loader_seq, device, epochs, lr=LEARNING_RATE)
                print(f"  F1: {metrics['F1']:.4f} | Train: {train_time:.2f}s | Params: {param_count:,}")
                seed_results.append({"Seed": seed, "Model": "1D-CNN", **metrics, "Train_Time": train_time, "Infer_Time": infer_time, "Memory_MB": mem, "Parameters": param_count, "FLOPs": flops})
                last_seed_preds["1D-CNN"] = (metrics["preds"], metrics["labels"])
                all_seed_preds["1D-CNN"].append((metrics["preds"], metrics["labels"]))
                model_histories["1D-CNN"]["losses"].append(metrics["epoch_losses"])
                model_histories["1D-CNN"]["stds"].append(metrics["epoch_stds"])
                model_histories["1D-CNN"]["val_accs"].append(metrics["epoch_val_accs"])
                model_histories["1D-CNN"]["val_f1s"].append(metrics["epoch_val_f1s"])
                model_histories["1D-CNN"]["val_rocs"].append(metrics["epoch_val_rocs"])
                del model; gc.collect(); torch.cuda.empty_cache()
                
            elif model_name == "CNN-LSTM":
                print("--- Training CNN-LSTM Baseline ---")
                model = CNNLSTM(in_channels=2, num_classes=num_classes).to(device)
                param_count = get_parameter_count(model)
                flops = estimate_flops(model, "sequence", device)
                metrics, train_time, infer_time, mem = train_and_eval_dl(model, "sequence", train_loader_seq, val_loader_seq, test_loader_seq, device, epochs, lr=LEARNING_RATE)
                print(f"  F1: {metrics['F1']:.4f} | Train: {train_time:.2f}s | Params: {param_count:,}")
                seed_results.append({"Seed": seed, "Model": "CNN-LSTM", **metrics, "Train_Time": train_time, "Infer_Time": infer_time, "Memory_MB": mem, "Parameters": param_count, "FLOPs": flops})
                last_seed_preds["CNN-LSTM"] = (metrics["preds"], metrics["labels"])
                all_seed_preds["CNN-LSTM"].append((metrics["preds"], metrics["labels"]))
                model_histories["CNN-LSTM"]["losses"].append(metrics["epoch_losses"])
                model_histories["CNN-LSTM"]["stds"].append(metrics["epoch_stds"])
                model_histories["CNN-LSTM"]["val_accs"].append(metrics["epoch_val_accs"])
                model_histories["CNN-LSTM"]["val_f1s"].append(metrics["epoch_val_f1s"])
                model_histories["CNN-LSTM"]["val_rocs"].append(metrics["epoch_val_rocs"])
                del model; gc.collect(); torch.cuda.empty_cache()
                
            elif model_name == "TCN":
                print("--- Training TCN Baseline ---")
                model = TCN(in_channels=2, num_classes=num_classes).to(device)
                param_count = get_parameter_count(model)
                flops = estimate_flops(model, "sequence", device)
                metrics, train_time, infer_time, mem = train_and_eval_dl(model, "sequence", train_loader_seq, val_loader_seq, test_loader_seq, device, epochs, lr=LEARNING_RATE)
                print(f"  F1: {metrics['F1']:.4f} | Train: {train_time:.2f}s | Params: {param_count:,}")
                seed_results.append({"Seed": seed, "Model": "TCN", **metrics, "Train_Time": train_time, "Infer_Time": infer_time, "Memory_MB": mem, "Parameters": param_count, "FLOPs": flops})
                last_seed_preds["TCN"] = (metrics["preds"], metrics["labels"])
                all_seed_preds["TCN"].append((metrics["preds"], metrics["labels"]))
                model_histories["TCN"]["losses"].append(metrics["epoch_losses"])
                model_histories["TCN"]["stds"].append(metrics["epoch_stds"])
                model_histories["TCN"]["val_accs"].append(metrics["epoch_val_accs"])
                model_histories["TCN"]["val_f1s"].append(metrics["epoch_val_f1s"])
                model_histories["TCN"]["val_rocs"].append(metrics["epoch_val_rocs"])
                del model; gc.collect(); torch.cuda.empty_cache()
                
            elif model_name == "Transformer":
                print("--- Training Transformer Baseline ---")
                model = TransformerClassifier(in_channels=2, num_classes=num_classes).to(device)
                param_count = get_parameter_count(model)
                flops = estimate_flops(model, "sequence", device)
                metrics, train_time, infer_time, mem = train_and_eval_dl(model, "sequence", train_loader_seq, val_loader_seq, test_loader_seq, device, epochs, lr=LEARNING_RATE)
                print(f"  F1: {metrics['F1']:.4f} | Train: {train_time:.2f}s | Params: {param_count:,}")
                seed_results.append({"Seed": seed, "Model": "Transformer", **metrics, "Train_Time": train_time, "Infer_Time": infer_time, "Memory_MB": mem, "Parameters": param_count, "FLOPs": flops})
                last_seed_preds["Transformer"] = (metrics["preds"], metrics["labels"])
                all_seed_preds["Transformer"].append((metrics["preds"], metrics["labels"]))
                model_histories["Transformer"]["losses"].append(metrics["epoch_losses"])
                model_histories["Transformer"]["stds"].append(metrics["epoch_stds"])
                model_histories["Transformer"]["val_accs"].append(metrics["epoch_val_accs"])
                model_histories["Transformer"]["val_f1s"].append(metrics["epoch_val_f1s"])
                model_histories["Transformer"]["val_rocs"].append(metrics["epoch_val_rocs"])
                del model; gc.collect(); torch.cuda.empty_cache()
                
            elif model_name == "GAT-No-Shapelets":
                print("--- Training GAT-No-Shapelets Baseline ---")
                model = GATWithoutShapelets(window_size=500, hidden_channels=HIDDEN_CHANNELS, out_channels=num_classes).to(device)
                param_count = get_parameter_count(model)
                flops = estimate_flops(model, "graph", device)
                metrics, train_time, infer_time, mem = train_and_eval_dl(model, "graph", train_loader_graph, val_loader_graph, test_loader_graph, device, epochs, lr=LEARNING_RATE)
                print(f"  F1: {metrics['F1']:.4f} | Train: {train_time:.2f}s | Params: {param_count:,}")
                seed_results.append({"Seed": seed, "Model": "GAT-No-Shapelets", **metrics, "Train_Time": train_time, "Infer_Time": infer_time, "Memory_MB": mem, "Parameters": param_count, "FLOPs": flops})
                last_seed_preds["GAT-No-Shapelets"] = (metrics["preds"], metrics["labels"])
                all_seed_preds["GAT-No-Shapelets"].append((metrics["preds"], metrics["labels"]))
                model_histories["GAT-No-Shapelets"]["losses"].append(metrics["epoch_losses"])
                model_histories["GAT-No-Shapelets"]["stds"].append(metrics["epoch_stds"])
                model_histories["GAT-No-Shapelets"]["val_accs"].append(metrics["epoch_val_accs"])
                model_histories["GAT-No-Shapelets"]["val_f1s"].append(metrics["epoch_val_f1s"])
                model_histories["GAT-No-Shapelets"]["val_rocs"].append(metrics["epoch_val_rocs"])
                del model; gc.collect(); torch.cuda.empty_cache()
                
            elif model_name == "GCN-No-Shapelets":
                print("--- Training GCN-No-Shapelets Baseline ---")
                model = GCNWithoutShapelets(window_size=500, hidden_channels=HIDDEN_CHANNELS, out_channels=num_classes).to(device)
                param_count = get_parameter_count(model)
                flops = estimate_flops(model, "graph", device)
                metrics, train_time, infer_time, mem = train_and_eval_dl(model, "graph", train_loader_graph, val_loader_graph, test_loader_graph, device, epochs, lr=LEARNING_RATE)
                print(f"  F1: {metrics['F1']:.4f} | Train: {train_time:.2f}s | Params: {param_count:,}")
                seed_results.append({"Seed": seed, "Model": "GCN-No-Shapelets", **metrics, "Train_Time": train_time, "Infer_Time": infer_time, "Memory_MB": mem, "Parameters": param_count, "FLOPs": flops})
                last_seed_preds["GCN-No-Shapelets"] = (metrics["preds"], metrics["labels"])
                all_seed_preds["GCN-No-Shapelets"].append((metrics["preds"], metrics["labels"]))
                model_histories["GCN-No-Shapelets"]["losses"].append(metrics["epoch_losses"])
                model_histories["GCN-No-Shapelets"]["stds"].append(metrics["epoch_stds"])
                model_histories["GCN-No-Shapelets"]["val_accs"].append(metrics["epoch_val_accs"])
                model_histories["GCN-No-Shapelets"]["val_f1s"].append(metrics["epoch_val_f1s"])
                model_histories["GCN-No-Shapelets"]["val_rocs"].append(metrics["epoch_val_rocs"])
                del model; gc.collect(); torch.cuda.empty_cache()
                
            elif model_name == "Random-Forest":
                print("--- Training Random Forest Baseline ---")
                reset_memory_tracker()
                start_feat_time = time.time()
                X_train_rf = extract_all_features(X_train_seq, use_multiprocessing=not args.debug)
                X_test_rf = extract_all_features(X_test_seq, use_multiprocessing=not args.debug)
                feat_time = time.time() - start_feat_time
                
                scaler = StandardScaler()
                X_train_rf = scaler.fit_transform(X_train_rf)
                X_test_rf = scaler.transform(X_test_rf)
                
                start_train_time = time.time()
                rf_model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_model.fit(X_train_rf, y_train_seq)
                train_time = (time.time() - start_train_time) + feat_time
                
                start_infer_time = time.time()
                y_pred = rf_model.predict(X_test_rf)
                y_prob = rf_model.predict_proba(X_test_rf)
                total_infer_time = time.time() - start_infer_time
                
                acc = accuracy_score(y_test_seq, y_pred)
                prec = precision_score(y_test_seq, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test_seq, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test_seq, y_pred, average='macro', zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_test_seq, y_prob, multi_class='ovr', average='macro')
                except Exception:
                    roc_auc = 0.5
                    
                infer_time_ms = (total_infer_time / len(y_test_seq)) * 1000.0
                mem = get_peak_memory()
                print(f"  F1: {f1:.4f} | Train: {train_time:.2f}s | Params: N/A")
                
                seed_results.append({
                    "Seed": seed, "Model": "Random-Forest", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
                    "ROC-AUC": roc_auc, "Train_Time": train_time, "Infer_Time": infer_time_ms, "Memory_MB": mem,
                    "Parameters": "N/A", "FLOPs": "N/A"
                })
                last_seed_preds["Random-Forest"] = (y_pred, y_test_seq)
                all_seed_preds["Random-Forest"].append((y_pred, y_test_seq))
                del rf_model; gc.collect()
                
        # -----------------------------------------------------------------------------
        # Generate model-specific plots, logs, and confusion matrices INSTANTANEOUSLY!
        # -----------------------------------------------------------------------------
        print(f"📁 [EXPORT] Instantly generating folders, figures and logs for: {model_name}")
        
        # 1. Standalone plots for the LATEST seed directly under model_dir
        if GENERATE_CONFUSION_MATRICES and model_name in all_seed_preds and len(all_seed_preds[model_name]) > 0:
            preds, labels = all_seed_preds[model_name][-1]
            plot_model_confusion_matrices(model_name, preds, labels, label_encoder.classes_, model_dir)
            
        if model_name != "Random-Forest" and model_name in model_histories and len(model_histories[model_name]["losses"]) > 0:
            losses = model_histories[model_name]["losses"][-1]
            stds = model_histories[model_name]["stds"][-1]
            val_accs = model_histories[model_name]["val_accs"][-1]
            plot_training_val_curves(model_name, losses, stds, val_accs, model_dir)
            
        # 2. Standalone summary for the LATEST seed directly under model_dir
        df_model_seeds = pd.DataFrame([res for res in seed_results if res["Model"] == model_name])
        if len(df_model_seeds) > 0:
            latest_seed_row = df_model_seeds.iloc[-1]
            latest_seed_val = latest_seed_row["Seed"]
            
            summary_content = []
            summary_content.append("=" * 65)
            summary_content.append(f"SCIENTIFIC PERFORMANCE & RESOURCE SUMMARY: {model_name} (Latest Run)")
            summary_content.append(f"Based on Seed: {latest_seed_val}")
            summary_content.append("=" * 65)
            summary_content.append("\n[PERFORMANCE METRICS]:")
            summary_content.append(f"  - Accuracy:     {latest_seed_row['Accuracy']*100:.2f}%")
            summary_content.append(f"  - Precision:    {latest_seed_row['Precision']*100:.2f}%")
            summary_content.append(f"  - Recall:       {latest_seed_row['Recall']*100:.2f}%")
            summary_content.append(f"  - F1-Score:     {latest_seed_row['F1']*100:.2f}%")
            summary_content.append(f"  - ROC-AUC:      {latest_seed_row['ROC-AUC']:.3f}")
            
            summary_content.append("\n[COMPUTATIONAL COMPLEXITY & RESOURCE EFFICIENCY]:")
            params = latest_seed_row["Parameters"]
            param_str = f"{params:,} parameters" if isinstance(params, (int, float)) else "N/A"
            summary_content.append(f"  - Parameter Count:    {param_str}")
            flops = latest_seed_row["FLOPs"]
            flops_str = f"{flops:,} FLOPs" if isinstance(flops, (int, float)) else "N/A (Non-DL/Unavailable)"
            summary_content.append(f"  - FLOPs Count:        {flops_str}")
            summary_content.append(f"  - Training Time:      {latest_seed_row['Train_Time']:.2f}s")
            summary_content.append(f"  - Inference Latency:  {latest_seed_row['Infer_Time']:.3f}ms per sample")
            summary_content.append(f"  - Memory Footprint:   {latest_seed_row['Memory_MB']:.2f} MB")
            summary_content.append("=" * 65)
            
            with open(os.path.join(model_dir, "metrics_summary.txt"), "w") as f:
                f.write("\n".join(summary_content))
            print(f"[EXPORT] Saved model performance summary (Latest Seed) to {model_name}/metrics_summary.txt")
            
            # 3. If multiple seeds, organize the individual seeds under seeds/ subdirectory
            if seeds_count > 1:
                model_seeds_dir = os.path.join(model_dir, "seeds")
                os.makedirs(model_seeds_dir, exist_ok=True)
                
                # A. Write the seed-averaged comparison file multi_seed_comparison.txt
                agg_data = {"Model": model_name, "Parameters": df_model_seeds.iloc[0]["Parameters"], "FLOPs": df_model_seeds.iloc[0]["FLOPs"]}
                for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train_Time", "Infer_Time", "Memory_MB"]:
                    vals = df_model_seeds[m].tolist()
                    agg_data[f"{m}_mean"] = np.mean(vals)
                    agg_data[f"{m}_ci"] = compute_95_ci(vals)
                    
                multi_seed_content = []
                multi_seed_content.append("=" * 65)
                multi_seed_content.append(f"MULTI-SEED SCIENTIFIC SUMMARY (AVERAGE ACROSS {seeds_count} SEEDS): {model_name}")
                multi_seed_content.append("=" * 65)
                multi_seed_content.append("\n[PERFORMANCE METRICS] (Mean ± 95% Confidence Interval):")
                multi_seed_content.append(f"  - Accuracy:     {agg_data['Accuracy_mean']*100:.2f}% ± {agg_data['Accuracy_ci']*100:.2f}%")
                multi_seed_content.append(f"  - Precision:    {agg_data['Precision_mean']*100:.2f}% ± {agg_data['Precision_ci']*100:.2f}%")
                multi_seed_content.append(f"  - Recall:       {agg_data['Recall_mean']*100:.2f}% ± {agg_data['Recall_ci']*100:.2f}%")
                multi_seed_content.append(f"  - F1-Score:     {agg_data['F1_mean']*100:.2f}% ± {agg_data['F1_ci']*100:.2f}%")
                multi_seed_content.append(f"  - ROC-AUC:      {agg_data['ROC-AUC_mean']:.3f} ± {agg_data['ROC-AUC_ci']:.3f}")
                
                multi_seed_content.append("\n[COMPUTATIONAL COMPLEXITY & RESOURCE EFFICIENCY]:")
                params = agg_data["Parameters"]
                param_str = f"{params:,} parameters" if isinstance(params, (int, float)) else "N/A"
                multi_seed_content.append(f"  - Parameter Count:    {param_str}")
                flops = agg_data["FLOPs"]
                flops_str = f"{flops:,} FLOPs" if isinstance(flops, (int, float)) else "N/A (Non-DL/Unavailable)"
                multi_seed_content.append(f"  - FLOPs Count:        {flops_str}")
                multi_seed_content.append(f"  - Training Time:      {agg_data['Train_Time_mean']:.2f}s ± {agg_data['Train_Time_ci']:.2f}s (Total)")
                multi_seed_content.append(f"  - Inference Latency:  {agg_data['Infer_Time_mean']:.3f}ms ± {agg_data['Infer_Time_ci']:.3f}ms per sample")
                multi_seed_content.append(f"  - Memory Footprint:   {agg_data['Memory_MB_mean']:.2f} MB ± {agg_data['Memory_MB_ci']:.2f} MB")
                multi_seed_content.append("=" * 65)
                
                with open(os.path.join(model_seeds_dir, "multi_seed_comparison.txt"), "w") as f:
                    f.write("\n".join(multi_seed_content))
                print(f"[EXPORT] Saved multi-seed aggregated summary to {model_name}/seeds/multi_seed_comparison.txt")
                
                # B. Save individual seed folders seed_[seed_val]/
                for idx, seed_val in enumerate(seeds_list):
                    seed_dir = os.path.join(model_seeds_dir, f"seed_{seed_val}")
                    os.makedirs(seed_dir, exist_ok=True)
                    
                    if GENERATE_CONFUSION_MATRICES and model_name in all_seed_preds and len(all_seed_preds[model_name]) > idx:
                        p, l = all_seed_preds[model_name][idx]
                        plot_model_confusion_matrices(model_name, p, l, label_encoder.classes_, seed_dir)
                        
                    if model_name != "Random-Forest" and model_name in model_histories and len(model_histories[model_name]["losses"]) > idx:
                        ls = model_histories[model_name]["losses"][idx]
                        ss = model_histories[model_name]["stds"][idx]
                        va = model_histories[model_name]["val_accs"][idx]
                        plot_training_val_curves(model_name, ls, ss, va, seed_dir)
                        
                    row = df_model_seeds.iloc[idx]
                    seed_content = []
                    seed_content.append("=" * 65)
                    seed_content.append(f"SCIENTIFIC PERFORMANCE & RESOURCE SUMMARY: {model_name} (Seed {seed_val})")
                    seed_content.append("=" * 65)
                    seed_content.append("\n[PERFORMANCE METRICS]:")
                    seed_content.append(f"  - Accuracy:     {row['Accuracy']*100:.2f}%")
                    seed_content.append(f"  - Precision:    {row['Precision']*100:.2f}%")
                    seed_content.append(f"  - Recall:       {row['Recall']*100:.2f}%")
                    seed_content.append(f"  - F1-Score:     {row['F1']*100:.2f}%")
                    seed_content.append(f"  - ROC-AUC:      {row['ROC-AUC']:.3f}")
                    
                    seed_content.append("\n[COMPUTATIONAL COMPLEXITY & RESOURCE EFFICIENCY]:")
                    params = row["Parameters"]
                    param_str = f"{params:,} parameters" if isinstance(params, (int, float)) else "N/A"
                    seed_content.append(f"  - Parameter Count:    {param_str}")
                    flops = row["FLOPs"]
                    flops_str = f"{flops:,} FLOPs" if isinstance(flops, (int, float)) else "N/A (Non-DL/Unavailable)"
                    seed_content.append(f"  - FLOPs Count:        {flops_str}")
                    seed_content.append(f"  - Training Time:      {row['Train_Time']:.2f}s")
                    seed_content.append(f"  - Inference Latency:  {row['Infer_Time']:.3f}ms per sample")
                    seed_content.append(f"  - Memory Footprint:   {row['Memory_MB']:.2f} MB")
                    seed_content.append("=" * 65)
                    
                    with open(os.path.join(seed_dir, "metrics_summary.txt"), "w") as f:
                        f.write("\n".join(seed_content))

    print("\n" + "=" * 60)
    print("📊 BENCHMARKING COMPLETE! SAVING OUTPUTS... 📊")
    print("=" * 60)
    
    # Structure Comparison Folder
    comp_dir = os.path.join(OUTPUT_FOLDER, "Comparison")
    os.makedirs(comp_dir, exist_ok=True)
    
    df_seeds = pd.DataFrame(seed_results)
    models_run = [m for m, active in toggles.items() if active]
    
    # -----------------------------------------------------------------------------
    # 1. Compile and save results for the LATEST SEED directly under Comparison/
    # -----------------------------------------------------------------------------
    print(f"📁 [EXPORT] Generating latest-seed comparison reports and plots directly in Figures/Comparison/")
    
    # Save raw seed-level results to Comparison directory for completeness
    df_seeds.to_csv(os.path.join(comp_dir, "results_per_seed.csv"), index=False)
    
    # Build df_agg_latest: last seed run's metrics for each model, setting CIs to 0.0
    latest_rows = []
    latest_seed_val = seeds_list[-1]
    for model_name in models_run:
        df_model_latest = df_seeds[(df_seeds["Model"] == model_name) & (df_seeds["Seed"] == latest_seed_val)]
        if len(df_model_latest) == 0:
            continue
        row = df_model_latest.iloc[0]
        agg_data = {"Model": model_name, "Parameters": row["Parameters"], "FLOPs": row["FLOPs"]}
        for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train_Time", "Infer_Time", "Memory_MB"]:
            agg_data[f"{m}_mean"] = row[m]
            agg_data[f"{m}_ci"] = 0.0
        latest_rows.append(agg_data)
    df_agg_latest = pd.DataFrame(latest_rows)
    
    df_agg_latest.to_csv(os.path.join(comp_dir, "results_aggregated.csv"), index=False)
    
    if GENERATE_COMPARISON_PLOTS:
        # Plot multi-line cumulative learning curves for latest seed run
        plot_cumulative_val_curves(model_histories, comp_dir, latest_only=True)
        # Save side-by-side performance bar plot (error-bar capsize=0 because CI=0.0)
        generate_plots(df_agg_latest, comp_dir)
        # Plot parameter complexities log-scale bar chart
        plot_model_complexities(df_agg_latest, comp_dir)
        # Plot separate resource comparisons (no error-bars because CI=0.0)
        generate_resource_comparison_plots(df_agg_latest, comp_dir)
        
    if GENERATE_LATEX_TABLE:
        dummy_p_values = {m: 1.0 for m in models_run if m != "Shapelet-GAT"}
        generate_latex_table(df_agg_latest, dummy_p_values, comp_dir)
        
    # Print and save the pretty unicode ASCII complexity table for the latest seed
    print_pretty_complexity_table(df_agg_latest, comp_dir)
    
    # Generate the detailed model complexity & performance comparison CSV for the latest seed
    create_detailed_comparison_csv(df_agg_latest, toggles, comp_dir)
    
    # Generate master comparative text log for the latest seed
    master_log_latest = []
    master_log_latest.append("=" * 125)
    master_log_latest.append(f"MASTER COMPARATIVE PERFORMANCE & COMPLEXITY REPORT (Latest Run - Seed {latest_seed_val})")
    master_log_latest.append("=" * 125)
    master_log_latest.append(f"{'Model Name':<20} | {'Accuracy':<10} | {'F1-Score':<10} | {'Params (K)':<10} | {'FLOPs':<12} | {'Train Time':<10} | {'Latency':<10} | {'Memory':<10}")
    master_log_latest.append("-" * 125)
    
    for idx, row in df_agg_latest.iterrows():
        model_name = row["Model"]
        params = row["Parameters"]
        params_k = f"{params/1000.0:.1f}k" if isinstance(params, (int, float)) else "N/A"
        flops = row["FLOPs"]
        flops_str = f"{flops:,}" if isinstance(flops, (int, float)) else "N/A"
        
        master_log_latest.append(
            f"{model_name:<20} | "
            f"{row['Accuracy_mean']*100:>8.2f}% | "
            f"{row['F1_mean']*100:>8.2f}% | "
            f"{params_k:>10} | "
            f"{flops_str:>12} | "
            f"{row['Train_Time_mean']:>8.2f}s | "
            f"{row['Infer_Time_mean']:>8.3f}ms | "
            f"{row['Memory_MB_mean']:>8.1f}MB"
        )
    master_log_latest.append("=" * 125)
    master_log_latest.append("\nNote: Standard deviations / CIs / p-values are not applicable for a single latest run.")
    master_log_latest.append("\n" + "=" * 125 + "\n")
    
    with open(os.path.join(comp_dir, "master_comparative_summary.txt"), "w") as f:
        f.write("\n".join(master_log_latest))
    print("[COMPARISON] Compiled latest seed master comparison report saved in Figures/Comparison/master_comparative_summary.txt")
    
    # -----------------------------------------------------------------------------
    # 2. If seeds_count > 1, save the aggregated comparisons under Comparison/seeds/
    # -----------------------------------------------------------------------------
    if seeds_count > 1:
        comp_seeds_dir = os.path.join(comp_dir, "seeds")
        os.makedirs(comp_seeds_dir, exist_ok=True)
        print(f"\n📁 [EXPORT] Generating multi-seed aggregated comparisons inside Figures/Comparison/seeds/")
        
        # Save a duplicate copy of raw seed-level results to seeds subfolder for convenience
        df_seeds.to_csv(os.path.join(comp_seeds_dir, "results_per_seed.csv"), index=False)
        
        # Build df_agg_multi: mean and 95% CI across all seeds
        agg_rows = []
        for model_name in models_run:
            df_model = df_seeds[df_seeds["Model"] == model_name]
            if len(df_model) == 0:
                continue
            agg_data = {"Model": model_name, "Parameters": df_model.iloc[0]["Parameters"], "FLOPs": df_model.iloc[0]["FLOPs"]}
            for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train_Time", "Infer_Time", "Memory_MB"]:
                vals = df_model[m].tolist()
                agg_data[f"{m}_mean"] = np.mean(vals)
                agg_data[f"{m}_ci"] = compute_95_ci(vals)
            agg_rows.append(agg_data)
        df_agg_multi = pd.DataFrame(agg_rows)
        
        df_agg_multi.to_csv(os.path.join(comp_seeds_dir, "results_aggregated.csv"), index=False)
        
        # Paired t-tests vs Proposed Shapelet-GAT across all seeds
        p_values_multi = {}
        if toggles["Shapelet-GAT"]:
            proposed_f1s = df_seeds[df_seeds["Model"] == "Shapelet-GAT"]["F1"].tolist()
            for model_name in models_run:
                if model_name == "Shapelet-GAT":
                    continue
                baseline_f1s = df_seeds[df_seeds["Model"] == model_name]["F1"].tolist()
                t_stat, p_val = run_statistical_test(proposed_f1s, baseline_f1s)
                p_values_multi[model_name] = p_val
                print(f"[STAT] [MULTI-SEED] Paired t-test vs {model_name} (p-value): {p_val:.4f}")
                
        if GENERATE_COMPARISON_PLOTS:
            # Plot multi-line cumulative learning curves (mean over seeds)
            plot_cumulative_val_curves(model_histories, comp_seeds_dir, latest_only=False)
            # Save side-by-side performance bar plot with 95% CI error bars
            generate_plots(df_agg_multi, comp_seeds_dir)
            # Plot parameter complexities log-scale bar chart
            plot_model_complexities(df_agg_multi, comp_seeds_dir)
            # Plot separate resource comparisons with 95% CI error bars
            generate_resource_comparison_plots(df_agg_multi, comp_seeds_dir)
            
        if GENERATE_LATEX_TABLE:
            generate_latex_table(df_agg_multi, p_values_multi, comp_seeds_dir)
            
        # Save pretty double-ruled ASCII table under Comparison/seeds/
        print_pretty_complexity_table(df_agg_multi, comp_seeds_dir)
        
        # Generate the detailed model complexity & performance comparison CSV under Comparison/seeds/
        create_detailed_comparison_csv(df_agg_multi, toggles, comp_seeds_dir)
        
        # Generate master comparative text log for multi-seed
        master_log_multi = []
        master_log_multi.append("=" * 125)
        master_log_multi.append(f"MASTER COMPARATIVE PERFORMANCE & COMPLEXITY REPORT (Averaged across {seeds_count} Seeds)")
        master_log_multi.append("=" * 125)
        master_log_multi.append(f"{'Model Name':<20} | {'Accuracy':<10} | {'F1-Score':<10} | {'Params (K)':<10} | {'FLOPs':<12} | {'Train Time':<10} | {'Latency':<10} | {'Memory':<10}")
        master_log_multi.append("-" * 125)
        
        for idx, row in df_agg_multi.iterrows():
            model_name = row["Model"]
            params = row["Parameters"]
            params_k = f"{params/1000.0:.1f}k" if isinstance(params, (int, float)) else "N/A"
            flops = row["FLOPs"]
            flops_str = f"{flops:,}" if isinstance(flops, (int, float)) else "N/A"
            
            master_log_multi.append(
                f"{model_name:<20} | "
                f"{row['Accuracy_mean']*100:>8.2f}% | "
                f"{row['F1_mean']*100:>8.2f}% | "
                f"{params_k:>10} | "
                f"{flops_str:>12} | "
                f"{row['Train_Time_mean']:>8.2f}s | "
                f"{row['Infer_Time_mean']:>8.3f}ms | "
                f"{row['Memory_MB_mean']:>8.1f}MB"
            )
        master_log_multi.append("=" * 125)
        
        if toggles["Shapelet-GAT"]:
            master_log_multi.append("\nStatistical Significance (Paired t-test on F1 vs Proposed Shapelet-GAT):")
            for model_name, p_val in p_values_multi.items():
                master_log_multi.append(f"  - vs {model_name:<20}: p-value = {p_val:.5f} (Significant: {p_val < 0.05})")
        master_log_multi.append("\n" + "=" * 125 + "\n")
        
        with open(os.path.join(comp_seeds_dir, "master_comparative_summary.txt"), "w") as f:
            f.write("\n".join(master_log_multi))
        print("[COMPARISON] Compiled multi-seed master comparison report saved in Figures/Comparison/seeds/master_comparative_summary.txt")
        
    print("\n✅ BENCHMARKING COMPLETED AND EXPORTED TO Figures/ DIRECTORY ✅\n")


if __name__ == "__main__":
    main()
