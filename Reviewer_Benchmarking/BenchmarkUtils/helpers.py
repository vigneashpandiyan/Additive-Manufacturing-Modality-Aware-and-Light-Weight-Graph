# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Computing parameter count, FLOPs estimates, peak memory footprints, and latency profiles.
- Training and evaluation execution loops for deep learning benchmarking.

Note: Any reuse of this code should be authorized by the code author.
"""

import time
import gc
import torch
import torch.nn as nn
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_peak_memory():
    """
    Description:
        Retrieves the maximum memory footprint occupied during execution. On CUDA machines, it queries GPU memory stats; otherwise, it queries system RAM RSS memory.
    Purpose:
        To profile peak hardware memory resource consumption.
    Input Types:
        - None
    Output Types:
        - peak_memory (float): Memory footprint in Megabytes (MB).
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    else:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)          # Convert to MB


def reset_memory_tracker():
    """
    Description:
        Clears cached GPU memory and resets the PyTorch maximum memory allocation statistics. Runs garbage collection.
    Purpose:
        To clear past allocation buffers before beginning a new model profiling run to avoid leaks/stale figures.
    Input Types:
        - None
    Output Types:
        - None
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def get_parameter_count(model):
    """
    Description:
        Iterates over the parameters of a model to calculate the total number of trainable parameter values.
    Purpose:
        To profile trainable parameter footprint of deep learning networks.
    Input Types:
        - model (torch.nn.Module): The target model to profile.
    Output Types:
        - param_count (int): Sum total of trainable scalar parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, model_type, device):
    """
    Description:
        Profiles the model by passing a simulated input batch matching the input shapes and uses the `thop` library to estimate total Floating Point Operations.
    Purpose:
        To estimate analytical runtime complexity (FLOPs) of DL architectures.
    Input Types:
        - model (torch.nn.Module): The network model.
        - model_type (str): The class format ('graph' or 'sequence').
        - device (torch.device): Computation target device.
    Output Types:
        - flops (int or str): Estimated total FLOPs count, or 'N/A' if an error occurs or `thop` is missing.
    """
    model.eval()
    try:
        import thop
        if model_type == "graph":
            num_nodes = 19
            x = torch.randn(num_nodes, 2, 500).to(device)
            src, dst = [], []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)
            batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
            
            flops, params = thop.profile(model, inputs=(x, edge_index, batch), verbose=False)
        else:
            x = torch.randn(1, 2, 5000).to(device)
            flops, params = thop.profile(model, inputs=(x,), verbose=False)
        return int(flops)
    except Exception as e:
        print(f"  [FLOPs] Warning: Could not compute FLOPs: {e}")
        return "N/A"


def train_and_eval_dl(model, model_type, train_loader, val_loader, test_loader, device, epochs, lr=0.005):
    """
    Description:
        Drives the standard deep learning pipeline. Iterates over specified epochs for sequence or graph training, calculates validation scores per epoch, tests the final trained model on test data, and tracks training speed, inference latency, and peak memory.
    Purpose:
        To train and score benchmarking architectures uniformly.
    Input Types:
        - model (torch.nn.Module): The network model to train.
        - model_type (str): Model category ('graph' or 'sequence').
        - train_loader (DataLoader): Training subset loader.
        - val_loader (DataLoader): Validation subset loader.
        - test_loader (DataLoader): Testing subset loader.
        - device (torch.device): target computational device.
        - epochs (int): Total epochs count.
        - lr (float): Learning rate float coefficient. Default is 0.005.
    Output Types:
        - tuple: (metrics, train_time, infer_time_per_sample_ms, peak_mem)
            - metrics (dict): Collection of accuracy, precision, recall, f1, roc-auc, lists of histories, and label outputs.
            - train_time (float): Training elapsed seconds.
            - infer_time_per_sample_ms (float): Inference latency in milliseconds per test sample.
            - peak_mem (float): Maximum memory footprint in Megabytes (MB).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    epoch_stds = []
    epoch_val_accs = []
    epoch_val_f1s = []
    epoch_val_rocs = []
    
    # 1. Training loop
    reset_memory_tracker()
    start_train_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == "graph":
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_loss = np.mean(batch_losses) if len(batch_losses) > 0 else 0.0
        std_loss = np.std(batch_losses) if len(batch_losses) > 0 else 0.0
        epoch_losses.append(avg_loss)
        epoch_stds.append(std_loss)
        
        # Validation metrics at the end of the epoch on val_loader
        model.eval()
        epoch_preds = []
        epoch_probs = []
        epoch_labels = []
        with torch.no_grad():
            for batch in val_loader:
                if model_type == "graph":
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    y = batch.y
                else:
                    x, y = batch
                    x = x.to(device)
                    out = model(x)
                    
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                epoch_preds.append(preds.cpu().numpy())
                epoch_probs.append(probs.cpu().numpy())
                epoch_labels.append(y.cpu().numpy())
                
        epoch_preds = np.concatenate(epoch_preds)
        epoch_probs = np.concatenate(epoch_probs)
        epoch_labels = np.concatenate(epoch_labels)
        
        val_acc = accuracy_score(epoch_labels, epoch_preds)
        val_f1 = f1_score(epoch_labels, epoch_preds, average='macro', zero_division=0)
        try:
            val_roc = roc_auc_score(epoch_labels, epoch_probs, multi_class='ovr', average='macro')
        except Exception:
            val_roc = 0.5
            
        epoch_val_accs.append(val_acc)
        epoch_val_f1s.append(val_f1)
        epoch_val_rocs.append(val_roc)
        
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"    [Epoch {epoch:03d}/{epochs:03d}] Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Val ROC-AUC: {val_roc:.4f}")
            
    train_time = time.time() - start_train_time
    
    # 2. Evaluation on test_loader
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    infer_start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "graph":
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                y = batch.y
            else:
                x, y = batch
                x = x.to(device)
                out = model(x)
                
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    total_infer_time = time.time() - infer_start_time
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Compute final test metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.5  # fallback
        
    peak_mem = get_peak_memory()
    infer_time_per_sample_ms = (total_infer_time / len(all_labels)) * 1000.0
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "preds": all_preds,
        "labels": all_labels,
        "epoch_losses": epoch_losses,
        "epoch_stds": epoch_stds,
        "epoch_val_accs": epoch_val_accs,
        "epoch_val_f1s": epoch_val_f1s,
        "epoch_val_rocs": epoch_val_rocs
    }
    
    return metrics, train_time, infer_time_per_sample_ms, peak_mem
