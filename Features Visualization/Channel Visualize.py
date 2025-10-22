import os
import numpy as np
from Utils_FFT import *  # Uses internal loop per sample, per channel
from Utils_STFT import Frequencyplot_STFT


# ==== Output Folder ====
base_dir = os.getcwd()
folder_created = os.path.join(base_dir, 'Visualization')
os.makedirs(folder_created, exist_ok=True)
print(f"[INFO] Output directory: {folder_created}")

# ==== Parameters ====
window_size = 5000
sampling_rate = 400000  # 0.4 MHz
time_vector = np.linspace(0, window_size / sampling_rate, window_size)
scales = np.arange(1, 128)


def normalize(data):
    """Normalize data to [-1, 1] range"""
    print("[NORMALIZATION] Performing Min-Max normalization to [-1, 1]")
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = 2 * ((data - data_min) / (data_max - data_min)) - 1
    return normalized


# ==== File Paths ====
D1_raw = r'C:\Cloud\GitLab\Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph\Data\D1_rawspace_5000.npy'
D2_raw = r'C:\Cloud\GitLab\Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph\Data\D2_rawspace_5000.npy'
classspace = r'C:\Cloud\GitLab\Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph\Data\classspace_5000.npy'

# ==== Load Data ====
try:
    X1 = np.load(D1_raw, allow_pickle=True)
    X2 = np.load(D2_raw, allow_pickle=True)
    Y = np.load(classspace, allow_pickle=True)
    if Y.ndim == 2:
        Y = Y.flatten()
    print(f"[DATA] Loaded: D1 shape = {X1.shape}, D2 shape = {
          X2.shape}, Labels shape = {Y.shape}")
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to load .npy files: {e}")

# ==== Class Mapping ====
class_labels = {
    1: "20%-Cu",
    2: "40%-Cu",
    3: "60%-Cu",
    4: "80%-Cu",
    5: "100%-Cu"
}

# ==== Select One Representative Sample per Class ====
classes = np.unique(Y)
print(f"[INFO] Classes found: {classes}")

np.random.seed(450)
selected_indices = [np.random.choice(np.where(Y == c)[0]) for c in classes]

X1_selected = X1[selected_indices]  # Channel 1
X2_selected = X2[selected_indices]  # Channel 2
Y_selected = Y[selected_indices]    # Labels
X1_selected = normalize(X1_selected)
X2_selected = normalize(X2_selected)
# ==== Create Combined Rawspace List: List of (CH1, CH2) ====
rawspace_list = list(zip(X1_selected, X2_selected))

# ==== Loop Over Channels Separately ====
for ch in [0, 1]:
    print(f"\n=== Processing Channel {ch+1} ===")
    Frequencyplot(
        rawspace_list=rawspace_list,
        labels_list=Y_selected,
        class_labels=class_labels,
        folder_created=folder_created,
        sample_rate=sampling_rate,
        windowsize=window_size,
        n_bands=5,
        channel=ch  # 0 = CH1, 1 = CH2
    )

    # Frequencyplot_STFT(
    #     rawspace_list=rawspace_list,
    #     labels_list=Y_selected,
    #     class_labels=class_labels,
    #     folder_created=folder_created,
    #     sample_rate=sampling_rate,
    #     windowsize=window_size,
    #     channel=ch  # 0 = CH1, 1 = CH2
    # )

    # Frequencyplot_(
    #     rawspace_list=rawspace_list,
    #     labels_list=Y_selected,
    #     class_labels=class_labels,
    #     folder_created=folder_created,
    #     sample_rate=sampling_rate,
    #     windowsize=window_size,
    #     n_bands=5,
    #     channel=ch  # 0 = CH1, 1 = CH2
    # )
