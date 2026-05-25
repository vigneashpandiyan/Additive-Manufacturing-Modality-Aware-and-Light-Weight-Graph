# -*- coding: utf-8 -*-
"""
DL execution, complexity, and resource tracking helpers.
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
    Get peak memory footprint (GPU allocated memory or CPU RSS memory in MB).
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    else:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)          # Convert to MB


def reset_memory_tracker():
    """
    Reset memory tracking.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def get_parameter_count(model):
    """
    Count the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, model_type, device):
    """
    Estimate FLOPs using the thop library.
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
    Trains and evaluates a deep learning model (sequence or graph).
    Returns:
        metrics: dict of test scores
        train_time: total training time
        infer_time: avg inference time per sample in ms
        peak_mem: peak memory in MB
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
