# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Feature engineering helpers for time-series baselines, including statistical moments and domain transformations.

Note: Any reuse of this code should be authorized by the code author.
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis, skew, entropy
from tqdm import tqdm
import multiprocessing as mp


def extract_channel_features(x):
    """
    Description:
        Extracts 22 hand-crafted statistical and spectral features (12 time-domain, 10 frequency-domain) from a single 1D signal.
    Purpose:
        To form tabular feature inputs for baseline machine learning models like Random Forest.
    Input Types:
        - x (numpy.ndarray): 1D array of single-channel sensor values.
    Output Types:
        - features (numpy.ndarray): 1D array containing all 22 extracted features.
    """
    features = []
    
    # 1. Time-domain features (12 features)
    mean_val = np.mean(x)
    std_val = np.std(x)
    rms_val = np.sqrt(np.mean(x**2))
    ptp_val = np.max(x) - np.min(x)
    kurt_val = kurtosis(x)
    skew_val = skew(x)
    
    abs_mean = np.mean(np.abs(x))
    crest_factor = np.max(np.abs(x)) / (rms_val + 1e-12)
    shape_factor = rms_val / (abs_mean + 1e-12)
    impulse_factor = np.max(np.abs(x)) / (abs_mean + 1e-12)
    
    # Margin factor: peak / (mean of sqrt(abs))**2
    sqrt_mean_sq = (np.mean(np.sqrt(np.abs(x))))**2
    margin_factor = np.max(np.abs(x)) / (sqrt_mean_sq + 1e-12)
    
    # Time-domain Shannon entropy of histogram
    hist_counts, _ = np.histogram(x, bins=50, density=True)
    time_entropy = entropy(hist_counts + 1e-12)
    
    energy_val = np.sum(x**2)
    
    features.extend([
        mean_val, std_val, rms_val, ptp_val, kurt_val, skew_val,
        crest_factor, shape_factor, impulse_factor, margin_factor,
        time_entropy, energy_val
    ])
    
    # 2. Frequency-domain features via Welch PSD (10 features)
    freqs, psd = signal.welch(x, fs=1.0, nperseg=256)
    psd_sum = np.sum(psd) + 1e-12
    
    # Spectral Centroid
    spec_centroid = np.sum(freqs * psd) / psd_sum
    
    # Spectral Spread (variance)
    spec_spread = np.sqrt(np.sum(((freqs - spec_centroid)**2) * psd) / psd_sum + 1e-12)
    
    # Spectral Skewness
    spec_skew = np.sum(((freqs - spec_centroid)**3) * psd) / ((spec_spread**3) * psd_sum + 1e-12)
    
    # Spectral Kurtosis
    spec_kurt = np.sum(((freqs - spec_centroid)**4) * psd) / ((spec_spread**4) * psd_sum + 1e-12)
    
    # Peak Frequency
    peak_freq = freqs[np.argmax(psd)]
    
    # Spectral Entropy
    spec_entropy = entropy(psd / psd_sum + 1e-12)
    
    features.extend([spec_centroid, spec_spread, spec_skew, spec_kurt, peak_freq, spec_entropy])
    
    # Relative power in 4 equal sub-bands
    num_bins = len(psd)
    band_len = num_bins // 4
    for b in range(4):
        start_idx = b * band_len
        end_idx = (b + 1) * band_len if b < 3 else num_bins
        band_power = np.sum(psd[start_idx:end_idx]) / psd_sum
        features.append(band_power)
        
    return np.array(features)


def extract_sample_features(sample):
    """
    Description:
        Extracts and concatenates tabular feature vectors from both sensor channels for a single sample.
    Purpose:
        To form a unified dual-channel feature vector representing the multi-modal input at a given timestep.
    Input Types:
        - sample (numpy.ndarray): 2D array of shape [2, T] containing channel 1 and channel 2 signals.
    Output Types:
        - concatenated_features (numpy.ndarray): 1D array of shape [44] with features from both channels.
    """
    ch1_feats = extract_channel_features(sample[0])
    ch2_feats = extract_channel_features(sample[1])
    return np.concatenate([ch1_feats, ch2_feats])


def extract_all_features(X_data, use_multiprocessing=True):
    """
    Description:
        Extracts tabular features for all samples in a batch. Supports parallel CPU processing via multiprocessing pool.
    Purpose:
        To perform fast batch feature engineering across the entire training/testing datasets for ML evaluation.
    Input Types:
        - X_data (numpy.ndarray): Sequence dataset array of shape [B, 2, T].
        - use_multiprocessing (bool): Enable or disable parallel pool execution. Default is True.
    Output Types:
        - feature_matrix (numpy.ndarray): 2D feature matrix of shape [B, 44].
    """
    print(f"[FEATURES] Extracting handcrafted features for {len(X_data)} samples...")
    
    if use_multiprocessing:
        # Use CPU multiprocessing to speed up feature extraction
        num_cores = max(1, mp.cpu_count() - 1)
        print(f"[FEATURES] Using {num_cores} CPU cores for parallel extraction...")
        with mp.Pool(num_cores) as pool:
            features_list = list(tqdm(pool.imap(extract_sample_features, X_data), total=len(X_data)))
    else:
        features_list = []
        for i in range(len(X_data)):
            features_list.append(extract_sample_features(X_data[i]))
            
    return np.array(features_list)
