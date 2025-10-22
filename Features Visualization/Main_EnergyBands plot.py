# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

This script is part of the publication:
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

# %% Plotting config
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

# %% Paths and folders
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print("[INFO] Base path:", total_path)

folder_name = 'Visualization'
path_ = os.path.join(total_path, folder_name)
os.makedirs(path_, exist_ok=True)
print("[INFO] Saving plots to:", path_)

# %% Load data
data_feature_path = os.path.join(path_, 'Feature_FFT_5000.npy')
data_class_path = os.path.join(path_, 'classspace_5000.npy')

data_feature = np.load(data_feature_path)
data_class = np.load(data_class_path).flatten()

print("[INFO] Feature shape:", data_feature.shape)
print("[INFO] Class shape:", data_class.shape)

# %% Map class labels
label_map = {
    1: '20%-Cu',
    2: '40%-Cu',
    3: '60%-Cu',
    4: '80%-Cu',
    5: '100%-Cu'
}
class_labels = np.vectorize(label_map.get)(data_class)

# %% Define frequency band names
feature_names = ['0-20 kHZ', '20-40 kHZ', '40-60 kHZ', '60-80 kHZ', '80-100 kHZ',
                 '100-120 kHZ', '120-140 kHZ', '140-160 kHZ', '160-180 kHZ', '180-200 kHZ']

# %% Create DataFrame
df_feature = pd.DataFrame(
    data_feature[:, -len(feature_names):], columns=feature_names)
df_feature['Categorical'] = class_labels

# %% Normalize feature values between 0 and 1 (across all bands globally)
features_only = df_feature[feature_names].values
labels_only = df_feature['Categorical'].values

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_only)

df_feature = pd.DataFrame(features_scaled, columns=feature_names)
df_feature['Categorical'] = labels_only

print("[INFO] Normalized all feature values between 0 and 1.")

# %% Balance the dataset
min_count = df_feature['Categorical'].value_counts().min()
# min_count = 1000
df_balanced = pd.concat([
    df_feature[df_feature['Categorical'] == cat].sample(
        min_count, random_state=42)
    for cat in df_feature['Categorical'].unique()
])
print("[INFO] Balanced class counts:\n",
      df_balanced['Categorical'].value_counts())

# %% Reshape for plotting
df_melted = df_balanced.melt(
    id_vars='Categorical', var_name='Frequency Band', value_name='Spectral Energy')

# %% Plotting function


def plot_frequency_distribution(data_melted, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    sns.set(font_scale=2)
    sns.set_style("whitegrid", {'axes.grid': False})

    # ✅ Define the custom order
    ordered_labels = ['20%-Cu', '40%-Cu', '60%-Cu', '80%-Cu', '100%-Cu']

    # ✅ Generate ordered color map using tab10
    cmap = plt.get_cmap('tab10')
    color_map = {
        label: cmap(i / len(ordered_labels))
        for i, label in enumerate(ordered_labels)
    }

    # ✅ Ensure Categorical column is a categorical type with order
    data_melted['Categorical'] = pd.Categorical(
        data_melted['Categorical'], categories=ordered_labels, ordered=True)

    # ✅ Remove the last frequency band group entirely
    data_melted = data_melted[data_melted['Frequency Band']
                              != data_melted['Frequency Band'].unique()[-1]]

    # Plot
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=data_melted,
        x="Spectral Energy",
        y="Frequency Band",
        hue="Categorical",
        hue_order=ordered_labels,
        palette=color_map,
        ci="sd"
    )

    # Titles and axis labels
    plt.title('Normalized spectral energy - Optical emission',
              fontsize=20, pad=20)
    plt.xlabel('Normalized spectral energy (a.u)', fontsize=22, labelpad=10)
    plt.ylabel('Frequency band', fontsize=22, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Legend on the right
    plt.legend(
        title='316L - Cu composition',
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=18,
        title_fontsize=20,
        frameon=False
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = os.path.join(save_path, "Normalized_Frequency_Distribution.png")
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    print(f"[✓] Plot saved to {filename}")


# def plot_frequency_distribution(data_melted, save_path):
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import os

#     sns.set(font_scale=2)
#     sns.set_style("whitegrid", {'axes.grid': False})

#     # ✅ Define the custom order
#     ordered_labels = ['20%-Cu', '40%-Cu', '60%-Cu', '80%-Cu', '100%-Cu']

#     # ✅ Generate ordered color map using Set2
#     cmap = plt.get_cmap('tab10')
#     color_map = {
#         label: cmap(i / len(ordered_labels))
#         for i, label in enumerate(ordered_labels)
#     }

#     # ✅ Ensure Categorical column is a categorical type with order
#     data_melted['Categorical'] = pd.Categorical(
#         data_melted['Categorical'], categories=ordered_labels, ordered=True)

#     # Increase figure width for space on the right
#     plt.figure(figsize=(14, 8))
#     ax = sns.barplot(
#         data=data_melted,
#         x="Spectral Energy",
#         y="Frequency Band",
#         hue="Categorical",
#         hue_order=ordered_labels,       # ✅ Enforce custom order
#         palette=color_map,
#         ci="sd"
#     )

#     # Titles and axis labels
#     plt.title('Normalized Power Spectral Density Distribution',
#               fontsize=20, pad=20)
#     plt.xlabel('Normalized Spectral Energy', fontsize=22, labelpad=10)
#     plt.ylabel('Frequency Band', fontsize=22, labelpad=10)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)

#     # Legend on the right
#     plt.legend(
#         title='316L - Cu composition',
#         loc='upper left',
#         bbox_to_anchor=(1.02, 1),
#         fontsize=18,
#         title_fontsize=20,
#         frameon=False
#     )

#     # Adjust layout to make room for legend
#     plt.tight_layout(rect=[0, 0, 0.85, 1])
#     filename = os.path.join(save_path, "Normalized_Frequency_Distribution.png")
#     plt.savefig(filename, dpi=400, bbox_inches='tight')
#     plt.show()
#     print(f"[✓] Plot saved to {filename}")
# %% Call the plot
plot_frequency_distribution(df_melted, path_)
