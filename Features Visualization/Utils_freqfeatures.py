# -*- coding: utf-8 -*-
"""
@author: vpsora
contact: vigneashwara.solairajapandiyan@utu.fi,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Adaptive In-situ Monitoring for Laser Powder Bed Fusion:Self-Supervised Learning for Layer Thickness Monitoring Across Scan lengths based on Pyrometry"

@any reuse of this code should be authorized by the code author
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
    Normalize an array to the range of -1 to 1.

    Args:
        array (numpy.ndarray): The input array to be normalized.

    Returns:
        numpy.ndarray: The normalized array.

    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = [
        ((x - min_val) / (max_val - min_val)) * 2 - 1 for x in array]
    normalized_array = np.array(normalized_array)

    return normalized_array


def get_band(band_size, band_max_size):
    """
    Generates a list of values representing frequency bands.

    Args:
        band_size (int): The number of bands to generate.
        band_max_size (float): The maximum size of the frequency band.

    Returns:
        list: A list of values representing the frequency bands.

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
    Calculate the delta power and relative power for a given power spectral density (psd) using the specified frequency bands.

    Args:
        psd (array-like): The power spectral density.
        band (array-like): The frequency bands.
        freqs (array-like): The frequencies corresponding to the power spectral density.
        band_size (int): The size of the frequency bands.

    Returns:
        tuple: A tuple containing two lists - Feature_deltapower and Feature_relativepower.
            - Feature_deltapower (list): The delta power for each frequency band.
            - Feature_relativepower (list): The relative power for each frequency band.

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

    # This function calculates the frequency features of a given input signal.
    # It takes three parameters: val (input signal), sample_rate (sampling rate of the signal), and band_size (number of frequency bands).
    # It returns a list of feature vectors.

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
    # This function calculates the frequency features of a given input signal.
    # It takes three parameters: data_new (input signal), sample_rate (sampling rate of the signal), and band_size (number of frequency bands).
    # It returns a list of feature vectors.
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
