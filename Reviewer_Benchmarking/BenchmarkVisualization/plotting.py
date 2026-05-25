# -*- coding: utf-8 -*-
"""
Visualization and plotting utilities.
Generates comparative performance plots, learning curves, and confusion matrices.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def generate_plots(aggregated_df, save_folder):
    """
    Generate professional, publication-quality bar plots comparing Accuracy and F1-Score
    of all models with 95% Confidence Interval error bars. Gridlines are explicitly disabled.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    models = aggregated_df["Model"].tolist()
    acc_means = aggregated_df["Accuracy_mean"].tolist()
    acc_cis = aggregated_df["Accuracy_ci"].tolist()
    f1_means = aggregated_df["F1_mean"].tolist()
    f1_cis = aggregated_df["F1_ci"].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    # Use clean default/white styling (no automatic grids)
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors_acc = ['#008080' if "Shapelet-GAT" in m else '#708090' for m in models]
    colors_f1 = ['#20B2AA' if "Shapelet-GAT" in m else '#A0B0C0' for m in models]
    
    ax.bar(x - width/2, acc_means, width, yerr=acc_cis, label='Accuracy', 
           color=colors_acc, edgecolor='black', capsize=4, alpha=0.9)
    ax.bar(x + width/2, f1_means, width, yerr=f1_cis, label='F1-Score (Macro)', 
           color=colors_f1, edgecolor='black', capsize=4, alpha=0.9)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Reviewer Benchmarking: Proposed vs Baselines', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, facecolor='white', edgecolor='black', fontsize=10)
    
    # Explicitly turn off grid lines
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "performance_comparison.png"), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_model_confusion_matrices(model_name, preds, labels, classes, save_folder):
    """
    Generate and save absolute and row-normalized confusion matrices for a given model.
    Saves the images directly into the specified model folder. Gridlines are explicitly disabled.
    """
    os.makedirs(save_folder, exist_ok=True)
    num_classes = len(classes)
    
    # Use clean default/white styling
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    
    # 1. Absolute confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.grid(False) # Turn off grid lines over matrix cells
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "confusion_matrix.png"))
    plt.show()
    plt.close()
    
    # 2. Normalized confusion matrix
    cm_norm = confusion_matrix(labels, preds, labels=list(range(num_classes)), normalize='true') * 100
    cm_norm = np.nan_to_num(cm_norm) # handle missing classes safely
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp_norm.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format=".1f")
    ax.grid(False) # Turn off grid lines over matrix cells
    plt.title(f"Confusion Matrix (%): {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "confusion_matrix_percent.png"))
    plt.show()
    plt.close()


def plot_training_val_curves(model_name, losses, stds, val_accs, save_folder):
    """
    Generate and save training loss and validation accuracy over epochs for a given model
    in a clean 1x2 layout. Shading represents standard deviation of batch losses for that epoch.
    All gridlines are explicitly disabled.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    losses = np.array(losses)
    stds = np.array(stds)
    val_accs = np.array(val_accs)
    
    num_epochs = len(losses)
    epochs = np.arange(1, num_epochs + 1)
    
    # Use clean default/white styling
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Plot 1: Training Loss (Mean + Std fill style across batches)
    ax1.plot(epochs, losses, color='#E74C3C', lw=2, label='Mean Loss')
    ax1.fill_between(epochs, losses - stds, losses + stds, color='#E74C3C', alpha=0.2, label='±1 Std Dev')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title(f'{model_name} Training Loss (Batch-level)', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', edgecolor='black')
    ax1.grid(False)
    
    # Plot 2: Validation Accuracy (validation accuracy)
    ax2.plot(epochs, val_accs, color='#2ECC71', lw=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title(f'{model_name} Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=True, facecolor='white', edgecolor='black')
    ax2.grid(False)
    
    plt.suptitle(f'{model_name} Learning Dynamics (Latest Run)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "training_val_curves.png"), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cumulative_val_curves(model_histories, save_folder, latest_only=False):
    """
    Generate and save multiple figures comparing mean Validation metrics over epochs 
    for all DL models side-by-side (Accuracy, F1-Score, and ROC-AUC).
    If latest_only is True, plots the latest seed run instead of the seed average.
    All gridlines are explicitly disabled.
    """
    os.makedirs(save_folder, exist_ok=True)
    if not model_histories:
        return
        
    color_map = {
        "Shapelet-GAT": "#E74C3C",        # Vibrant Red
        "1D-CNN": "#3498DB",              # Blue
        "CNN-LSTM": "#9B59B6",            # Purple
        "TCN": "#F1C40F",                 # Yellow
        "Transformer": "#1ABC9C",         # Teal
        "GAT-No-Shapelets": "#E67E22",     # Orange
        "GCN-No-Shapelets": "#34495E"      # Charcoal/Slate
    }
    
    metrics_to_plot = [
        {"key": "val_accs", "label": "Validation Accuracy", "filename": "cumulative_accuracy_comparison.png", "y_min": -0.02},
        {"key": "val_f1s", "label": "Validation F1-Score (Macro)", "filename": "cumulative_f1_comparison.png", "y_min": -0.02},
        {"key": "val_rocs", "label": "Validation ROC-AUC", "filename": "cumulative_roc_auc_comparison.png", "y_min": 0.48}
    ]
    
    for metric in metrics_to_plot:
        plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        has_data = False
        for model_name, history in model_histories.items():
            if model_name == "Random-Forest" or metric["key"] not in history or not history[metric["key"]]:
                continue
                
            vals = np.array(history[metric["key"]]) # Shape: [num_seeds, num_epochs]
            num_epochs = vals.shape[1]
            epochs = np.arange(1, num_epochs + 1)
            
            if latest_only:
                val_to_plot = vals[-1, :]
            else:
                val_to_plot = np.mean(vals, axis=0)
            
            color = color_map.get(model_name, "#7F8C8D")
            lw = 2.5 if "Shapelet" in model_name else 1.8
            ls = "-" if "Shapelet" in model_name else "--"
            
            ax.plot(epochs, val_to_plot, color=color, lw=lw, ls=ls, label=model_name)
            has_data = True
            
        if not has_data:
            plt.close()
            continue
            
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric["label"], fontsize=12, fontweight='bold')
        title_suffix = " (Latest Run)" if latest_only else " (Average across Seeds)"
        ax.set_title(f'Cumulative {metric["label"]} Comparison{title_suffix}', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(metric["y_min"], 1.02)
        ax.legend(frameon=True, facecolor='white', edgecolor='black', loc='lower right', fontsize=10)
        ax.grid(False) # Ensure no grid lines
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, metric["filename"]), bbox_inches='tight')
        plt.show()
        plt.close()


def plot_model_complexities(aggregated_df, save_folder):
    """
    Generate a professional bar plot comparing the number of trainable parameters
    across all deep learning models. Gridlines are explicitly disabled.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # Filter out models with "N/A" parameters (like Random Forest)
    df_dl = aggregated_df[aggregated_df["Parameters"] != "N/A"].copy()
    if len(df_dl) == 0:
        return
        
    models = df_dl["Model"].tolist()
    params = [float(p) for p in df_dl["Parameters"]]
    
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = ['#008080' if "Shapelet-GAT" in m else '#708090' for m in models]
    
    x = np.arange(len(models))
    bars = ax.bar(x, params, width=0.5, color=colors, edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Trainable Parameters (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity Comparison: Trainable Parameters', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_yscale('log') # Use log scale because sizes differ by orders of magnitude (e.g. 3K vs 115K)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "model_parameter_comparison.png"), bbox_inches='tight')
    plt.show()
    plt.close()


def generate_resource_comparison_plots(aggregated_df, save_folder):
    """
    Generate separate, publication-quality bar plots comparing FLOPs count, Training Time,
    Inference Latency, and Memory Footprint across all active models with 95% CI error bars.
    Gridlines are explicitly disabled.
    """
    os.makedirs(save_folder, exist_ok=True)
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    
    # 1. FLOPs Count Comparison (Deep Learning models only)
    df_dl = aggregated_df[aggregated_df["FLOPs"] != "N/A"].copy()
    if len(df_dl) > 0:
        models = df_dl["Model"].tolist()
        flops = [float(f) for f in df_dl["FLOPs"]]
        colors = ['#008080' if "Shapelet-GAT" in m else '#708090' for m in models]
        x = np.arange(len(models))
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bars = ax.bar(x, flops, width=0.5, color=colors, edgecolor='black', alpha=0.9)
        ax.set_ylabel('FLOPs Count (Log Scale)', fontsize=12, fontweight='bold')
        ax.set_title('Computational Complexity Comparison: FLOPs Count', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_yscale('log')
        
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "model_flops_comparison.png"), bbox_inches='tight')
        plt.show()
        plt.close()
        
    # All models (including Random Forest) for resource metrics
    models_all = aggregated_df["Model"].tolist()
    colors_all = ['#008080' if "Shapelet-GAT" in m else '#708090' for m in models_all]
    x_all = np.arange(len(models_all))
    
    # 2. Training Time Comparison (Mean ± 95% CI)
    train_means = aggregated_df["Train_Time_mean"].tolist()
    train_cis = aggregated_df["Train_Time_ci"].tolist()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(x_all, train_means, width=0.5, yerr=train_cis, color=colors_all, edgecolor='black', capsize=4, alpha=0.9)
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Complexity Comparison: Total Training Time', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_all)
    ax.set_xticklabels(models_all, rotation=45, ha='right', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "model_training_time_comparison.png"), bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 3. Inference Latency Comparison (Mean ± 95% CI)
    infer_means = aggregated_df["Infer_Time_mean"].tolist()
    infer_cis = aggregated_df["Infer_Time_ci"].tolist()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(x_all, infer_means, width=0.5, yerr=infer_cis, color=colors_all, edgecolor='black', capsize=4, alpha=0.9)
    ax.set_ylabel('Inference Latency (ms per sample)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Latency Comparison: Average Latency per Sample', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_all)
    ax.set_xticklabels(models_all, rotation=45, ha='right', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "model_inference_latency_comparison.png"), bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 4. Memory Footprint Comparison (Mean ± 95% CI)
    mem_means = aggregated_df["Memory_MB_mean"].tolist()
    mem_cis = aggregated_df["Memory_MB_ci"].tolist()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(x_all, mem_means, width=0.5, yerr=mem_cis, color=colors_all, edgecolor='black', capsize=4, alpha=0.9)
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Complexity Comparison: Peak Memory Footprint', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_all)
    ax.set_xticklabels(models_all, rotation=45, ha='right', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "model_memory_footprint_comparison.png"), bbox_inches='tight')
    plt.show()
    plt.close()
