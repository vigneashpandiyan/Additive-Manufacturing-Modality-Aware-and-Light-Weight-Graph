# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Normalizing input signals into range [-1, 1].
- Partitioning absolute frequency spectra into dynamic frequency bands.
- Computing absolute delta power and relative power ratio features using Welch's/Periodogram methods.

Note: Any reuse of this code should be authorized by the code author.
"""
# %%
# Libraries to import
import numpy as np
import pandas as pd
import pywt
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from scipy.signal import welch, periodogram
from numpy.fft import fftshift, fft
from scipy.signal import find_peaks
import statistics
from scipy import stats
from collections import Counter
from scipy.stats import entropy
from scipy.signal import hilbert, chirp
from scipy.stats import entropy


def normalize_to_minus_one(array):
    """
    Description:
        Normalizes the elements of an input array to the range [-1, 1] using min-max scaling.
    Purpose:
        To scale signals/waveforms within a standard range to allow consistent comparison.
    Input Types:
        - array (numpy.ndarray or list): The input array to be normalized.
    Output Types:
        - normalized_array (numpy.ndarray): The normalized array scaled to [-1, 1].
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = [
        ((x - min_val) / (max_val - min_val)) * 2 - 1 for x in array]
    normalized_array = np.array(normalized_array)

    return normalized_array


def get_band(band_size, band_max_size):
    """
    Description:
        Generates a sequence of frequency values dividing the range from 0 to band_max_size into band_size sub-bands.
    Purpose:
        To establish frequency boundary intervals for power spectral calculations.
    Input Types:
        - band_size (int): Number of sub-bands to generate.
        - band_max_size (float or int): Upper boundary of the total frequency range.
    Output Types:
        - band (list): A list of starting frequencies for each sub-band.
    """
    band_window = 0
    band = []
    for y in range(band_size):
        print(y)
        band.append(band_window)
        band_window += band_max_size / band_size
    print(band)
    return band


def spectrumpower(psd, band, freqs, band_size):
    """
    Description:
        Computes the absolute delta power and relative power ratios for a signal across specified frequency bands.
    Purpose:
        To extract quantitative energy-based features from the power spectral density.
    Input Types:
        - psd (array-like): Power spectral density values of the signal.
        - band (array-like): Frequencies representing sub-band boundaries.
        - freqs (array-like): The frequencies corresponding to the PSD.
        - band_size (int): Total number of sub-bands.
    Output Types:
        - Feature_deltapower (list): Absolute delta power for each band.
        - Feature_relativepower (list): Relative power ratio for each band.
    """
    length = len(band)
    # print(length)
    Feature_deltapower = []
    Feature_relativepower = []
    for i in range(band_size-1):
        if i <= (len(band)):
            ii = i
            low = band[ii]
            ii = i+1
            high = band[ii]
            idx_delta = np.logical_and(freqs >= low, freqs <= high)
            total_power = sum(psd)
            delta_power = sum(psd[idx_delta])
            delta_rel_power = delta_power / total_power
            Feature_deltapower.append(delta_power)
            Feature_relativepower.append(delta_rel_power)

    return Feature_deltapower, Feature_relativepower
# %%


def function_freq(val, sample_rate, band_size):
    """
    Description:
        Computes power spectral density and calculates delta and relative power features for a single 1D signal sequence.
    Purpose:
        To serve as a core feature-extraction mapping from raw signal sequence to frequency domain feature vectors.
    Input Types:
        - val (array-like): 1D array representing raw temporal signal values.
        - sample_rate (int): Signal sampling rate in Hz.
        - band_size (int): Number of frequency bands.
    Output Types:
        - Feature_vectors (numpy.ndarray): Concatenated array of delta and relative power features.
    """

    i = 0
    win = 4 * sample_rate
    freqs, psd = periodogram(val, sample_rate, window='hamming')
    band_max_size = sample_rate//2 + 20000
    band = get_band(band_size, band_max_size)

    print(band)

    Feature1, Feature2 = spectrumpower(psd, band, freqs, band_size)
    Feature1 = np.asarray(Feature1)
    Feature2 = np.asarray(Feature2)

    print("Feature1:", Feature1.shape)
    print("Feature1:", Feature2.shape)

    Feature = np.concatenate((Feature1, Feature2))

    if i == 0:
        #     print("--reached")
        size_of_Feature_vectors = int(len(Feature))

        Feature_vectors = np.empty((0, size_of_Feature_vectors))

    # print(label)
    Feature_vectors = np.append(Feature_vectors, [Feature], axis=0)

    return Feature_vectors

# %%


def Freqfunction(data_new, sample_rate, band_size):
    """
    Description:
        Processes a collection of signals, extracting frequency features row by row or channel by channel.
    Purpose:
        To perform batch frequency feature extraction across all samples in the raw dataset.
    Input Types:
        - data_new (numpy.ndarray): Multi-dimensional signal matrix of shape [window_length, num_samples].
        - sample_rate (int): Sampling rate of the raw signals in Hz.
        - band_size (int): Number of frequency sub-bands to generate.
    Output Types:
        - featurelist (list): List of frequency feature vectors for all samples.
    """
    columnsdata = data_new.transpose()
    columns = np.atleast_2d(columnsdata).shape[1]
    featurelist = []
    classlist = []
    rawlist = []

    # for row in loop:
    for k in range(columns):

        val = columnsdata[:, k]
        Feature_vectors = function_freq(val, sample_rate, band_size)
        print(k)
        for item in Feature_vectors:

            featurelist.append(item)

    return featurelist
