"""
    Functions for preparing dataset for model training, validation and testing.
"""

import os
from tqdm import tqdm
import numpy as np
from src.duel_result_preprocessing import load_csv_to_df, preprocess_duel, remove_duplicates


def split_dataset(x, y, ratio, shuffle=False, shuffle_seed=None):
    "Split dataset into train, validation and test sets."
    if shuffle:
        np.random.seed(shuffle_seed)
        indices = np.random.permutation(len(y))
        x = [xi[indices] for xi in x]
        y = y[indices]
        print("Dataset shuffled. Random seed: ", shuffle_seed)

    num_samples = len(y)
    train_end = int(num_samples * ratio[0])
    val_end = int(num_samples * (ratio[0] + ratio[1]))

    x_train, x_val, x_test = [[xi[:train_end] for xi in x], [xi[train_end:val_end] for xi in x],
                              [xi[val_end:] for xi in x]]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    print("Dataset split into train, validation and test sets. Split ratio: ", ratio)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def prepare_dataset(
        img_size,
        duel_results_path,
        image_file_dir,
        split_ratio=None,
        shuffle=False,
        shuffle_seed=None
):
    """
    Prepare dataset for model training, validation and testing.

    Args:
        img_size (int):
            The size of the image.
        duel_results_path (str):
            Path to duel results csv file.
        image_file_dir (str):
            Path to image file directory.
        split_ratio (list):
            List of ratios for splitting the dataset.
        shuffle (bool):
            Whether to shuffle the dataset.
        shuffle_seed (int):
            Random seed for shuffling the dataset.

    Returns:
        following format:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_dataset(...)
    """

    # Initialize duel image pairs and labels
    image_pairs = []
    labels = []

    # Load duel results from csv
    duel_results = load_csv_to_df(duel_results_path)
    duel_results = remove_duplicates(duel_results)

    # Get image names from duel results
    image1_names = duel_results['IMG1'].tolist()
    image2_names = duel_results['IMG2'].tolist()
    image_file_paths = [os.path.join(image_file_dir, image_name) for image_name in os.listdir(image_file_dir)]

    # Preprocess duels
    _ = 0  # Counter of duels
    with tqdm(total=len(image1_names), desc="Processing Duels") as pbar:
        # Loop through duel results
        for image1_name, image2_name in zip(image1_names, image2_names):
            image1_path = None
            image2_path = None

            for image_file_path in image_file_paths:
                if image1_name in image_file_path:
                    image1_path = image_file_path
                    continue
                if image2_name in image_file_path:
                    image2_path = image_file_path
                    continue

            if image1_path is None or image2_path is None:
                raise ValueError(f"Image not found: {image1_name} or {image2_name}.")

            duel_image_pair, duel_label = preprocess_duel(
                img_size,
                image1_path,
                image2_path,
                duel_results.iloc[_]['LABEL']
            )
            image_pairs.append(duel_image_pair)
            labels.append(duel_label)

            pbar.update(1)
            _ += 1

    image1_arrays = np.array([duel_image_pair[0] for duel_image_pair in image_pairs])
    image2_arrays = np.array([duel_image_pair[1] for duel_image_pair in image_pairs])
    label_arrays = np.array(labels)

    x = image1_arrays, image2_arrays
    y = label_arrays

    # Split the dataset
    if split_ratio is not None:
        return split_dataset(x, y, split_ratio, shuffle, shuffle_seed)

    else:
        split_ratio = [0.6, 0.2]  # 60% train, 20% validation and the rest for test
        return split_dataset(x, y, split_ratio, shuffle, shuffle_seed)

# def dataset_generator(dataset, batch_size):
#     num_samples = len(dataset['y'])
#     while True:
#         indices = np.random.permutation(num_samples)
#         for i in range(0, num_samples, batch_size):
#             batch_indices = indices[i:i + batch_size]
#             batch_images1 = dataset['x'][0][batch_indices]
#             batch_images2 = dataset['x'][1][batch_indices]
#             batch_labels = dataset['y'][batch_indices]
#             # data augmentation here
#             yield [batch_images1, batch_images2], batch_labels
