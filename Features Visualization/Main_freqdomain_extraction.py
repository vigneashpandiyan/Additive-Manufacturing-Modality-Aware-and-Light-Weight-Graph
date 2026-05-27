# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Loading acoustic emission raw dataset (D2) and label space.
- Normalizing raw acoustic waveforms to [-1, 1] range.
- Running spectral/frequency feature extraction on multiple overlapping windows using FFT-based relative power bands.
- Saving FFT feature matrices and labels for subsequent classification and visualization.

Note: Any reuse of this code should be authorized by the code author.
"""
# %%
# Libraries to import
import numpy as np
import scipy.signal as signal
from Utils_freqfeatures import *
import os
import pandas as pd

print(np.__version__)

# %%
# Get the path of the current working directory
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print("Working base path:", total_path)

# === Parameters ===
folder_name = 'Visualization'
sample_rate = 400000
windowsize = 5000
N = windowsize
t0 = 0
dt = 1 / sample_rate
time = np.arange(0, N) * dt + t0

# Define frequency band size if not defined inside Freqfunction
band_size = 11  # Example value; modify as needed

# %%
# Create a folder to save the data
path_ = os.path.join(total_path, folder_name)
print("Saving outputs to folder:", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

# %%
# Load dataset
dataset_path = r'C:\Cloud\GitLab\Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph\Data'
dataset_name = 'D2_rawspace_5000.npy'
dataset_label = 'classspace_5000.npy'

print("Loading dataset from:", dataset_path)
rawspace = np.load(os.path.join(dataset_path, dataset_name))
classspace = np.load(os.path.join(dataset_path, dataset_label))

# Normalize the dataset
rawspace = normalize_to_minus_one(rawspace)

# Extract features in frequency domain
featurespace = Freqfunction(rawspace, sample_rate, band_size)
featurespace = np.squeeze(np.asarray(featurespace))

# === Save FFT feature space ===
featurefile = os.path.join(path_, f'Feature_FFT_{windowsize}.npy')
np.save(featurefile, featurespace, allow_pickle=True)

# === Save class labels ===
classfile = os.path.join(path_, f'classspace_{windowsize}.npy')
np.save(classfile, classspace, allow_pickle=True)

print("\n✔ Frequency feature file saved:", featurefile)
print("✔ Class labels saved:", classfile)
