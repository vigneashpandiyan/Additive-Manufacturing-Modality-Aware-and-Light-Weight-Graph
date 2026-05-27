# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Computing Short-Time Fourier Transforms (STFT) on raw sensor window waveforms.
- Plotting and saving spectrogram energy responses for various PBF-LB composition levels.

Note: Any reuse of this code should be authorized by the code author.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal


def Frequencyplot_STFT(rawspace_list, labels_list, class_labels,
                       folder_created, sample_rate, windowsize, channel=0):
    """
    Description:
        Generates and saves a grid of Short-Time Fourier Transform (STFT) spectrogram plots for each alloy composition class, centering the last subplot if odd.
    Purpose:
        To examine temporal-spectral characteristics of raw sensory signals at different alloy composition ratios.
    Input Types:
        - rawspace_list (list): List of (CH1, CH2) signal tuples.
        - labels_list (list): Integer composition label array.
        - class_labels (dict): Integer-to-string composition mapping dict.
        - folder_created (str): Directory where the output spectrogram figure will be saved.
        - sample_rate (int): Frequency sampling rate in Hz.
        - windowsize (int): Placeholder parameter for legacy signature compatibility.
        - channel (int): Selected channel index (0 for Acoustic, 1 for Optical).
    Output Types:
        - save_path (str): File path of the saved PNG figure.
    """
    assert channel in [0, 1], "Channel must be 0 (CH1) or 1 (CH2)"
    if not os.path.exists(folder_created):
        os.makedirs(folder_created)

    unique_labels = sorted(set(labels_list))
    n = len(unique_labels)
    n_cols = 2
    n_rows = (n + 1) // n_cols

    cutoff = sample_rate // 2
    normalized_cutoff = min(cutoff / (sample_rate / 10), 0.99)
    b, a = signal.butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.3)

    pcm = None  # to hold the last plot's pcolormesh (for colorbar)

    for i, class_idx in enumerate(unique_labels):
        indices = [j for j, y in enumerate(labels_list) if y == class_idx]
        x = rawspace_list[indices[0]][channel]
        filtered_signal = signal.filtfilt(b, a, x)

        f, t, Zxx = signal.stft(filtered_signal, fs=sample_rate, nperseg=128)
        power = np.abs(Zxx) ** 2

        if i < n - 1 or n % 2 == 0:
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        else:
            ax = fig.add_subplot(gs[-1, :])
            box = ax.get_position()
            full_width = box.width
            desired_width = full_width * 0.45
            x_center = box.x0 + box.width / 2
            new_x0 = x_center - desired_width / 2
            ax.set_position([new_x0, box.y0, desired_width, box.height])

        pcm = ax.pcolormesh(t, f, power, shading='gouraud', cmap='viridis')
        ax.set_title(f"Class: {class_labels[class_idx]}", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_ylim(0, cutoff)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.tick_params(labelsize=10)
        ax.set_ylim(0, sample_rate // 10)

    # === Common colorbar ===
    if pcm:
        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Power', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f"{'Acoustic' if channel ==
                 0 else 'Optical'} Emission — STFT Spectrograms", fontsize=16)

    save_path = os.path.join(
        folder_created, f"STFT_Spectrograms_Channel{channel + 1}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return save_path
