# Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, Explainable Graph-Attention Sensor Fusion Framework

This repository contains the complete implementation and benchmarking suite for the manuscript:
**"Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, Explainable Graph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"**

Author: **vpsora**  
Contact: vigneashwara.solairajapandiyan@utu.fi | vigneashpandiyan@gmail.com  

---

## 📖 Framework Overview

This framework utilizes a novel **Shapelet-based Graph Attention Network (GAT)** to perform in-situ composition monitoring of graded multi-material alloys (316L Stainless Steel and CuCrZr) manufactured via laser powder bed fusion (PBF-LB). 

By extracting sliding temporal windows from co-axial optical and acoustic emissions, representing them as graph nodes, and learning discriminative shapelet matching distances, the network dynamically maps multi-modal signatures into a cohesive, explainable spatial attention graph.

### 🔬 1. Experimental Setup
![Experimental Setup](Data/Figure%201.jpg)
*Figure 1: In-situ multi-modal sensor monitoring experimental setup for the PBF-LB manufacturing process.*

### 🕸️ 2. Graph Connection
![Graph Connection](Data/Figure%202.jpg)
*Figure 2: Structure of signal segment sequence nodes and the temporal-spatial graph connections.*

### 💻 3. Code Pipeline
![Code Pipeline](Data/Figure%203.jpg)
*Figure 3: End-to-end code pipeline representing the dual-modality feature extraction, learnable shapelet networks, graph attention convolutions (GAT), pooling, and alloy composition estimation.*

---

## 📁 Repository Directory Structure

```directory
├── Features Visualization/      # Frequency-domain spectral features & signal diagnostics
├── Reviewer_Benchmarking/       # Comprehensive baseline models & LaTeX helper tools
├── Sensor fusion/               # Core multi-modal GAT-Shapelet framework code
├── Unimodal/                    # Single-modality baselines (Optical or Acoustic alone)
└── Data/                        # [Input Target] Holds raw .npy signal spaces and labels
```

---

## ⚙️ Folder Contents & Script Implementations

### 1. 🧠 `Sensor fusion/` (Core Multi-Modal Framework)
This is the main component of the project, orchestrating the multi-modal sensor fusion pipeline.

*   **`Config.py`**:
    *   *Objective*: Establishes global reproducibility parameters (random seeds), directories, dataset targets, validation sizes, and hyperparameters (`batch_size=256`, `num_epochs=300`, `shapelet_len=50`, `num_shapelets=10`).
*   **`Main.py`**:
    *   *Objective*: Orchestrates the end-to-end framework. Initializes environments, standardizes datasets, structures graph representations, trains/tests the model, saves checkpoints, and runs visual explanation utilities.
*   **`Main_Shapelet_Optimization.py`**:
    *   *Objective*: Executes a systematic sweep over shapelet counts ($k \in [2, 4, ..., 16]$) to profile model accuracy against parameter count, execution times, and computational FLOPs.
*   **`dataloader/data_loader.py`**:
    *   *Objective*: Implements stratified resampling to balance target compositions, standardizes sensors, and packages multi-channel signals into structured PyTorch Geometric `Data` graphs.
*   **`network/Network.py`**:
    *   *Objective*: Implements the GNN architecture:
        *   `BatchedShapeletExtractor`: A parallel learnable shapelet layer mapping signal segments to node features.
        *   `GNNWithAttention`: Fuses optical and acoustic node features using a 3-layer GAT structure, global average pooling, and classifier MLP.
*   **`trainer/Trainer.py`**:
    *   *Objective*: Handles core execution loops, parameter tracking, validation callbacks, checkpoint saves, and outputs training loss/validation curves.
*   **`utils/Inference.py` & `Utils.py`**:
    *   *Objective*: Performs saliency-based gradient backpropagation to evaluate channel/node contributions, and manages standardized GPU seed initializations.
*   **`visualization/Graph.py` & `Visualization.py`**:
    *   *Objective*: Generates high-fidelity explanatory figures: attention heatmaps, spatial spring layout graphs, PCA/t-SNE/UMAP dimension embeddings, and shapelet activation curves.

---

### 2. 🎛️ `Reviewer_Benchmarking/` (Benchmarking Suite)
Developed to benchmark the proposed framework against alternative deep learning sequence models.

*   **`Run_benchmarks.py`**:
    *   *Objective*: Executes repeated training sweeps across multiple architectures, logging test scores, latencies, memory footprints, and parameters.
*   **`BenchmarkDataloader/data_loader.py` & `features.py`**:
    *   *Objective*: Restructures raw signals into 1D sequences and sequences with statistical moments tailored for conventional architectures.
*   **`BenchmarkNetwork/`**:
    *   *Objective*: Houses baseline architectures:
        *   `cnn_1d.py` / `cnn_lstm.py`: Spatial and temporal convolutional baselines.
        *   `tcn.py` / `transformer.py`: Causal temporal convolutions and self-attention sequence models.
        *   `gcn_no_shapelets.py` / `gat_no_shapelets.py`: Graph GCN/GAT architectures using direct statistical features instead of learnable shapelets.
        *   `shapelet_gat.py`: Standard learnable shapelet attention baseline.
*   **`BenchmarkUtils/helpers.py`, `statistics.py`, & `latex_table.py`**:
    *   *Objective*: Calculates Peak FLOPs/Latency profiles, evaluates statistical confidence intervals (95% CI), performs Wilcoxon rank-sum tests, and formats benchmarking tables directly into LaTeX.
*   **`BenchmarkVisualization/plotting.py`**:
    *   *Objective*: Renders comparative accuracy vs complexity plots and benchmarking performance heatmaps.

---

### 3. 📻 `Unimodal/` (Single-Modality Baselines)
Evaluates GAT-Shapelet performance on individual modalities alone to quantify the advantages of sensor fusion.

*   **`Shaplet_Unimodal.py`**:
    *   *Objective*: Evaluates performance exclusively using either Optical emission (D1) or Acoustic emission (D2) signals.
*   **`trainer.py`**:
    *   *Objective*: Provides single-modality adapters for `BatchedShapeletExtractor` and `GNNWithAttention`.
*   **`utils.py`**:
    *   *Objective*: Manages data pre-scaling, graph translation from 1D waveforms, and dataset stats logging.

---

### 4. 📈 `Features Visualization/` (Signal Processing & Extraction)
Explores raw sensor behaviors and generates baseline diagnostic plots.

*   **`Channel Visualize.py`**:
    *   *Objective*: Extracts and compares optical/acoustic frequencies across all composition percentages.
*   **`Main_EnergyBands plot.py`**:
    *   *Objective*: Computes global relative power distributions and maps them to composition categories using Welch periodograms.
*   **`Main_freqdomain_extraction.py`**:
    *   *Objective*: Translates raw acoustic time series into sliding frequency-band relative power features.
*   **`Utils_FFT.py`, `Utils_freqfeatures.py`, & `Utils_STFT.py`**:
    *   *Objective*: Implements Fast Fourier Transforms, Short-Time Fourier Transforms (Spectrograms), and Welch periodograms for relative spectral band calculations.
*   **`untitled0.py`**:
    *   *Objective*: Simulates powder spread densities and plots dynamic multi-grid KDE overlays for layer-wise print analysis.

---

## 🚀 Execution & Verification

### 📋 Prerequisites
Set up your python environment (tested on Python 3.13 via Anaconda) and install PyTorch with CUDA support and PyTorch Geometric:
```bash
pip install numpy pandas matplotlib networkx scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/pub/whl/cu121
pip install torch-geometric
pip install thop  # Optional: for FLOPs profiling
```

### 🏃 Running the Core Sensor Fusion Pipeline
To run the primary GAT-Shapelet fusion model:
```bash
cd "Sensor fusion"
python Main.py
```

### 📊 Running the Parameter Optimization Sweep
To evaluate accuracy, latency, and resource metrics over shapelet counts $k \in [2, ..., 16]$:
```bash
cd "Sensor fusion"
python Main_Shapelet_Optimization.py
```

### ⚖️ Running Benchmarks
To compare against other architectures:
```bash
cd "Reviewer_Benchmarking"
python Run_benchmarks.py
```

---

## 🔒 Citation & Copyright

If you use this work, please cite:

**Title:** Learning composition-sensitive signatures in multi-material PBF-LB: a lightweight, modality-aware, explainable graph-attention sensor fusion framework for in-situ monitoring of graded 316L–CuCrZr alloys
**DOI:** [10.1007/s40964-026-01854-x](https://doi.org/10.1007/s40964-026-01854-x)
**Article Type:** Research
**Journal:** Progress in Additive Manufacturing

**Authors:**
Vigneashwara Pandiyan<sup>a,*</sup>, Antonios Baganis<sup>b,c</sup>, Antti Salminen<sup>a</sup>, Christian Leinenbach<sup>b,c</sup>

*   <sup>a</sup> Digital Manufacturing and Surface Engineering (DMS), Department of Mechanical and Materials Engineering, University of Turku, FI-20014, Turun yliopisto, Finland.
*   <sup>b</sup> Laboratory for Advanced Materials Processing (LAMP), Swiss Federal Laboratories for Materials Science and Technology (Empa)-CH-3602 Thun, Switzerland.
*   <sup>c</sup> Laboratory for Photonic Materials and Characterization, Ecole Polytechnique Fédérale de Lausanne, 1015 Lausanne, Switzerland.

### Citation Format (BibTeX)
```bibtex
@article{pandiyan2026learning,
  title={Learning composition-sensitive signatures in multi-material PBF-LB: a lightweight, modality-aware, explainable graph-attention sensor fusion framework for in-situ monitoring of graded 316L--CuCrZr alloys},
  author={Pandiyan, Vigneashwara and Baganis, Antonios and Salminen, Antti and Leinenbach, Christian},
  journal={Progress in Additive Manufacturing},
  year={2026},
  doi={10.1007/s40964-026-01854-x},
  publisher={Springer}
}
```

Any reuse of this code should be authorized by the code author:
**vpsora** (vigneashwara.solairajapandiyan@utu.fi | vigneashpandiyan@gmail.com).

