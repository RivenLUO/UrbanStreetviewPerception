import numpy as np

from keras import preprocessing


def normalize_min_max(array):
    # Calculate per-channel min and max values
    min_values = np.min(array, axis=0, keepdims=True)
    max_values = np.max(array, axis=0, keepdims=True)

    # Apply channel-wise min-max normalization
    normalized_array = (array - min_values) / (max_values - min_values)

    return normalized_array


def normalize_z_score(array):
    # Calculate per-channel mean and std values
    mean_values = np.mean(array, axis=0, keepdims=True)
    std_values = np.std(array, axis=0, keepdims=True)

    # Apply channel-wise z-score normalization
    normalized_array = (array - mean_values) / std_values

    return normalized_array
