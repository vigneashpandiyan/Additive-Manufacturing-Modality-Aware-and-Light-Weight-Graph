# Reviewer Benchmarking Pipeline

This directory contains a complete, isolated benchmarking pipeline designed to evaluate the proposed **Shapelet–GAT** model against seven diverse baseline models. 

The baseline models are:
1. **1D CNN** (Sequence-based)
2. **CNN–LSTM** (Sequence-based)
3. **TCN** (Temporal Convolutional Network, Sequence-based)
4. **Transformer** (Sequence-based)
5. **GAT without shapelets** (Graph-based baseline)
6. **GCN without shapelets** (Graph-based baseline)
7. **Random Forest** with 44 handcrafted time- and frequency-domain features (Standard machine learning baseline)

All neural network models share equivalent downstream classifier structures to ensure a scientifically rigorous and fair comparison.

---

## Environment Setup & Requirements
The pipeline has been verified on this machine with the local Anaconda environment:
* **Python Path**: `C:\Users\vpsora\AppData\Local\anaconda3\python.exe`
* **GPU Accelerator**: `NVIDIA RTX PRO 4000 Blackwell` (CUDA Available: True)
* **Required Libraries**: `torch`, `torch-geometric`, `thop` (for FLOPs), `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `psutil`.

---

## How to Run

To run a fast, end-to-end dry-run (verifies compilation, loading, feature extraction, and reporting with 1 epoch and 200 samples):
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"; & "C:\Users\vpsora\AppData\Local\anaconda3\python.exe" run_benchmarks.py --debug
```

To run the complete full benchmarking suite (default: 3 seeds, 30 epochs):
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"; & "C:\Users\vpsora\AppData\Local\anaconda3\python.exe" run_benchmarks.py
```

### Model-specific flags (True / False)
You can selectively enable or disable individual models by passing `--model_name False` or `--model_name True`. By default, all models are set to `True`.

For example, to run only the Proposed Shapelet-GAT and the Random Forest baseline:
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"; & "C:\Users\vpsora\AppData\Local\anaconda3\python.exe" run_benchmarks.py --cnn_1d False --cnn_lstm False --tcn False --transformer False --gat_no_shapelets False --gcn_no_shapelets False
```

### Script CLI Arguments:
* `--seeds`: Number of random seeds to run (default: `3`).
* `--epochs`: Number of deep learning training epochs (default: `30`).
* `--batch_size`: Batch size for DataLoader (default: `256`).
* `--debug`: Toggles quick debug mode (1 epoch, slices dataset to 200 samples, runs instantly).

---

## Outputs & Deliverables
All results and figures are exported to the `Reviewer_Benchmarking/outputs/` directory:
1. `results_per_seed.csv`: Performance metrics per model for each seed run.
2. `results_aggregated.csv`: Aggregated metrics (Mean $\pm$ 95% Confidence Interval), training/inference times, and memory/FLOPs consumption.
3. `performance_comparison.png`: Professional bar chart of F1 and Accuracy with 95% CI error bars.
4. `manuscript_table.tex`: Ready-to-use LaTeX code for your paper.
5. `confusion_matrices/`: ABS and Normalized Confusion Matrix plots for all 8 models.
