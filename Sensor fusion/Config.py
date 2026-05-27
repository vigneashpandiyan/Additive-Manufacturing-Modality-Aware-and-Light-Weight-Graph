# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Establishing default hyper-parameters: random seed, batch size, learning rates, epochs, test-val ratios, shapelet size, and directory targets.

Note: Any reuse of this code should be authorized by the code author.
"""

import os


# === System and Reproducibility Settings ===
SEED = 42
"""int: Random seed for all random number generators to ensure reproducibility.
    
Affects:
- NumPy random operations
- PyTorch random operations
- Python built-in random module
- Data shuffling/splitting operations
"""


# === Model Training Hyperparameters ===
num_epochs = 300
"""int: Number of complete passes through the training dataset.

Note:
    Lower values (3-10) are typical for early experimentation
    Higher values (50-100) may be needed for final training
"""

batch_size = 256
"""int: Number of samples processed before model parameters are updated.

Typical range:
- Small datasets: 32-128
- Medium datasets: 128-512
- Large datasets: 512-2048
"""


# === Data Processing Parameters ===
test_size = 0.20
"""float: Proportion of dataset to reserve for testing (0.0 to 1.0).

Note:
    Maintains class distribution through stratified splitting
    Typical values: 0.2-0.3 (20-30% for testing)
"""

val_size = 0.15
"""float: Proportion of dataset to reserve for validation (0.0 to 1.0).

Note:
    Ensures validation metrics track generalized model behavior.
"""

shapelet_len = 50
"""int: Length of learned shapelets in time steps.

Determines:
- Size of convolutional filters for shapelet learning
- Length of extracted subsequences
- Complexity of temporal patterns captured
"""

num_shapelets = 10
"""int: Number of shapelets to learn per input channel.

Affects:
- Model capacity (more shapelets = more complex patterns can be learned)
- Computational requirements
- Interpretability (fewer shapelets are easier to analyze)
"""


# === Directory and Path Configuration ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""str: Absolute path to the project root directory.

Used as base for constructing all other paths.
"""

data_folder = os.path.join(base_dir, "Data")
"""str: Path to directory containing all input data files.

Expected contents:
- D1_rawspace_5000.npy: Optical emission time series
- D2_rawspace_5000.npy: Acoustic emission time series
- classspace_5000.npy: Corresponding class labels
"""

plot_folder = os.path.join(base_dir, 'Sensor fusion', 'Figures', 'Full Run')
"""str: Path to directory for saving all visualization outputs.

Directory structure:
- Figures/
  - Dummy_check/
    - shapelet_visualizations/
    - attention_maps/
    - embeddings/
"""

# Create necessary directories if they don't exist
os.makedirs(plot_folder, exist_ok=True)
"""Ensures the plot output directory exists before saving any figures.

Note:
    exist_ok=True prevents errors if directory already exists
"""
