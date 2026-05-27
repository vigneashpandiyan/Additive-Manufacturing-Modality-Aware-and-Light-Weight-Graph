# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Computing Fast Fourier Transforms (FFT) on multi-channel sensors.
- Calculating power spectral density (PSD) and grouping frequencies into discrete bands.
- Plotting relative and absolute spectral energy across composition classes for optical and acoustic signals.

Note: Any reuse of this code should be authorized by the code author.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from matplotlib.gridspec import GridSpec


def Frequencyplot(rawspace_list, labels_list, class_labels,
                  folder_created, sample_rate, windowsize,
                  n_bands=5, channel=0):
    """
    Description:
        Generates a grid of Power Spectral Density (PSD) plots for each alloy composition class, centering the last subplot if the number of classes is odd.
    Purpose:
        To perform frequency domain analysis on multi-modal signals and verify spectral energy distributions per class.
    Input Types:
        - rawspace_list (list): List of tuples (CH1, CH2) where each element is a 1D numpy array of signal values.
        - labels_list (list): Corresponding class labels per sample.
        - class_labels (dict): Mapping from class index (int) to readable class name (str).
        - folder_created (str): Directory where the output figure is saved.
        - sample_rate (int): Sampling rate of the raw signals in Hz.
        - windowsize (int): Placeholder for legacy code compatibility.
        - n_bands (int): Number of frequency bands to shade under the PSD curve.
        - channel (int): Index of the channel to plot (0 for Acoustic, 1 for Optical).
    Output Types:
        - save_path (str): File path of the saved PNG figure.
    """
    assert channel in [0, 1], "Channel must be 0 (CH1) or 1 (CH2)"
    if not os.path.exists(folder_created):
        os.makedirs(folder_created)

    # ---- Class bookkeeping ----
    unique_labels = sorted(set(labels_list))
    n_classes = len(unique_labels)
    n_cols = 2
    n_rows = (n_classes + 1) // n_cols

    # ---- Filtering (anti-alias guard only; normalized to Nyquist) ----
    nyq = sample_rate / 2.0
    normalized_cutoff = min(nyq / nyq, 0.99)  # ~0.99 of Nyquist
    b, a = signal.butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)

    # ---- Bands in kHz for plotting & labels ----
    cutoff_hz = nyq
    cutoff_khz = cutoff_hz / 1000.0
    band_edges_khz = np.linspace(0.0, cutoff_khz, n_bands + 1)
    default_colors = ['blue', 'red', 'green', 'purple', 'orange']

    # ---- Pre-scan for a common y-limit (max PSD across classes) ----
    psd_max_list = []
    for class_idx in unique_labels:
        # take the first sample of that class
        indices = [j for j, y in enumerate(labels_list) if y == class_idx]
        x = rawspace_list[indices[0]][channel]
        x_filt = signal.filtfilt(b, a, x)

        # choose a safe segment length
        nperseg = int(min(4 * sample_rate, len(x_filt)))
        if nperseg < 16:
            # too short to compute a meaningful PSD
            psd_max_list.append(0.0)
            continue

        _, psd = signal.welch(x_filt, fs=sample_rate, nperseg=nperseg)
        psd_max_list.append(float(np.nanmax(psd)) if psd.size else 0.0)

    y_max = max(psd_max_list) if psd_max_list else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    # ---- Figure/layout ----
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.35)

    last_ax = None
    for i, class_idx in enumerate(unique_labels):
        # pick axis
        if i < n_classes - 1 or n_classes % 2 == 0:
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        else:
            # center the final single axis on the last row
            ax = fig.add_subplot(gs[-1, :])
            box = ax.get_position()
            full_width = box.width
            desired_width = full_width * 0.45
            x_center = box.x0 + box.width / 2
            new_x0 = x_center - desired_width / 2
            ax.set_position([new_x0, box.y0, desired_width, box.height])

        # data for this class (first sample)
        indices = [j for j, y in enumerate(labels_list) if y == class_idx]
        x = rawspace_list[indices[0]][channel]
        x_filt = signal.filtfilt(b, a, x)
        nperseg = int(min(4 * sample_rate, len(x_filt)))
        freqs_hz, psd = signal.welch(x_filt, fs=sample_rate, nperseg=nperseg)
        psd = psd * 1000.0  # Convert from V²/Hz → V²/kHz

        # convert X to kHz for plotting
        freqs_khz = freqs_hz / 1000.0

        # main trace
        ax.plot(freqs_khz, psd, color='k', lw=0.2)

        # shaded bands (in kHz)
        for j in range(n_bands):
            start_khz = band_edges_khz[j]
            end_khz = band_edges_khz[j + 1]
            color = default_colors[j % len(default_colors)]
            idx_band = (freqs_khz >= start_khz) & (freqs_khz <= end_khz)
            if np.any(idx_band):
                ax.fill_between(freqs_khz, psd, where=idx_band,
                                color=color, alpha=0.5,
                                label=f"{int(np.floor(start_khz))}–{int(np.floor(end_khz))} kHz")

        # formatting
        ax.set_ylim(0, y_max * 1000.0)
        ax.set_title(f"Class: {class_labels[class_idx]}", fontsize=18)
        ax.set_xlabel("Frequency (kHz)", fontsize=16)
        ax.set_ylabel("Spectral density \n (V²/kHz)", fontsize=16)
        # y in sci notation; x normal (kHz)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.tick_params(labelsize=16)
        ax.yaxis.get_offset_text().set_fontsize(16)

        last_ax = ax

    # ---- Shared legend (bands) ----
    if last_ax is not None:
        handles, labels = last_ax.get_legend_handles_labels()
        # Deduplicate while preserving order
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels)
                if not (l in seen or seen.add(l))]
        if uniq:
            uh, ul = zip(*uniq)
            fig.legend(uh, ul, loc='lower center', ncol=n_bands, fontsize=16)

    # ---- Title & save ----
    chan_name = "Acoustic" if channel == 0 else "Optical"
    fig.suptitle(
        f"{chan_name} emission — Frequency distribution", fontsize=18, y=0.98)

    save_path = os.path.join(
        folder_created, f"Welch_PSDs_Channel{channel + 1}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return save_path

# def Frequencyplot(rawspace_list, labels_list, class_labels,
#                   folder_created, sample_rate, windowsize, n_bands=5, channel=0):
#     """
#     Generate a grid of PSD plots with last subplot centered if odd number of classes.

#     Parameters:
#     - rawspace_list: List of (CH1, CH2) tuples
#     - labels_list: List of integer class labels
#     - class_labels: Dict mapping label → name
#     - folder_created: Directory to save the figure
#     - sample_rate: Sampling rate in Hz
#     - windowsize: For matplotlib performance
#     - n_bands: Number of frequency bands to shade
#     - channel: 0 = CH1 (Acoustic), 1 = CH2 (Optical)
#     """
#     assert channel in [0, 1], "Channel must be 0 (CH1) or 1 (CH2)"
#     if not os.path.exists(folder_created):
#         os.makedirs(folder_created)

#     # Prepare class-wise data
#     unique_labels = sorted(set(labels_list))
#     n = len(unique_labels)
#     n_cols = 2
#     n_rows = (n + 1) // n_cols
#     cutoff = sample_rate // 2
#     normalized_cutoff = min(cutoff / (sample_rate / 2), 0.99)
#     b, a = signal.butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)
#     band_edges = np.linspace(0, cutoff, n_bands + 1).astype(int)
#     default_colors = ['blue', 'red', 'green', 'purple', 'orange']

#     # Figure and layout
#     fig = plt.figure(figsize=(10, 9))
#     gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.15)

#     psd_max_list = []
#     for class_idx in unique_labels:
#         indices = [j for j, y in enumerate(labels_list) if y == class_idx]
#         x = rawspace_list[indices[0]][channel]
#         filtered_signal = signal.filtfilt(b, a, x)
#         win = 4 * sample_rate
#         _, psd = signal.welch(filtered_signal, fs=sample_rate, nperseg=win)
#         psd_max_list.append(np.max(psd))
#         y_max = max(psd_max_list)

#     for i, class_idx in enumerate(unique_labels):
#         indices = [j for j, y in enumerate(labels_list) if y == class_idx]
#         x = rawspace_list[indices[0]][channel]  # First sample of that class
#         filtered_signal = signal.filtfilt(b, a, x)
#         win = 4 * sample_rate
#         freqs, psd = signal.welch(filtered_signal, fs=sample_rate, nperseg=win)

#         if i < n - 1 or n % 2 == 0:
#             ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
#         else:
#             ax = fig.add_subplot(gs[-1, :])
#             box = ax.get_position()
#             full_width = box.width
#             desired_width = full_width * 0.45
#             x_center = box.x0 + box.width / 2
#             new_x0 = x_center - desired_width / 2
#             ax.set_position([new_x0, box.y0, desired_width, box.height])

#         ax.plot(freqs, psd, color='k', lw=0.2)
#         for j in range(n_bands):
#             start = band_edges[j]
#             end = band_edges[j + 1]
#             color = default_colors[j % len(default_colors)]
#             idx_band = np.logical_and(freqs >= start, freqs <= end)
#             ax.fill_between(freqs, psd, where=idx_band, color=color,
#                             alpha=0.5, label=f"{start//1000}–{end//1000} kHz")
#             ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#             ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#         ax.set_ylim(0, y_max)
#         ax.set_title(f"Class: {class_labels[class_idx]}", fontsize=18)
#         ax.set_xlabel("Frequency (Hz)", fontsize=12)
#         ax.set_ylabel("Spectral density", fontsize=16)
#         ax.tick_params(labelsize=16)

#     # Shared Legend
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=n_bands, fontsize=12)
#     fig.suptitle(f"{'Acoustic' if channel ==
#                  0 else 'Optical'} emission — Frequency distribution", fontsize=16)

#     save_path = os.path.join(
#         folder_created, f"Welch_PSDs_Channel{channel + 1}.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
#     return save_path


def Frequencyplot_(rawspace_list, labels_list, class_labels,
                   folder_created, sample_rate, windowsize, n_bands=5, channel=0):
    """
    Description:
        Loops over each sample to individually compute, shade frequency bands, and save high-resolution PSD plots for a selected channel.
    Purpose:
        To examine individual sensor sample spectral contents in detail.
    Input Types:
        - rawspace_list (list): List of tuples (CH1, CH2) representing dual-channel inputs.
        - labels_list (list or numpy.ndarray): Class label for each sample.
        - class_labels (dict): Mapping of integer class index to its descriptive name string.
        - folder_created (str): Output folder path.
        - sample_rate (int): Sampling rate in Hz.
        - windowsize (int): Used to configure matplotlib path chunking for rendering performance.
        - n_bands (int): Number of frequency bands to shade in the plot.
        - channel (int): Index of the channel (0 for Acoustic, 1 for Optical).
    Output Types:
        - None: Directly saves and renders the figures.
    """

    assert channel in [0, 1], "Channel must be 0 (Acoustic) or 1 (Optical)"
    channel_label = "Acoustic emission" if channel == 0 else "Optical emission"

    cutoff = sample_rate // 2
    normalized_cutoff = min(cutoff / (sample_rate / 2), 0.99)

    # Butterworth low-pass filter
    b, a = signal.butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)

    for idx, (multi_ch_signal, y) in enumerate(zip(rawspace_list, labels_list)):
        class_name = class_labels[y]
        label_base = f"{class_name}_Sample{idx}"
        print(f"[PROCESSING] {label_base} — {channel_label}")

        # === Extract selected channel ===
        signal_raw = multi_ch_signal[channel]

        # === Filter and PSD ===
        filtered_signal = signal.filtfilt(b, a, signal_raw)
        win = 4 * sample_rate
        freqs, psd = signal.welch(filtered_signal, fs=sample_rate, nperseg=win)

        # === Plot ===
        sns.set(style='white', font_scale=1.2)
        fig, ax = plt.subplots(figsize=(12, 7))
        plt.rcParams['agg.path.chunksize'] = windowsize
        ax.plot(freqs, psd, color='k', lw=0.6)

        # === Shade frequency bands ===
        band_edges = np.linspace(0, cutoff, n_bands + 1).astype(int)
        default_colors = ['blue', 'red', 'green',
                          'purple', 'orange', 'cyan', 'yellow']
        for i in range(n_bands):
            start = band_edges[i]
            end = band_edges[i + 1]
            color = default_colors[i % len(default_colors)]
            label = f"{start // 1000}–{end // 1000} kHz"
            idx_band = np.logical_and(freqs >= start, freqs <= end)
            ax.fill_between(freqs, psd, where=idx_band,
                            color=color, alpha=0.6, label=label)

        # === Formatting ===
        ax.set_xlabel('Frequency (Hz)', fontsize=20)
        ax.set_ylabel('Power Spectral Density', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=14)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(16)
        ax.xaxis.offsetText.set_fontsize(16)
        plt.ylim([0, psd.max() * 1.1])
        plt.title(f'Class: {class_name} — {
                  channel_label} — Welch PSD', fontsize=18)

        # === Save Plot ===
        file_label = channel_label.replace(" ", "_")
        save_path = os.path.join(folder_created, f"{label_base}_{
                                 file_label}_Frequency.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"[SAVED] {save_path}")

        # === Cleanup ===
        plt.show()
        plt.clf()
        plt.close(fig)
        del freqs, psd, filtered_signal
