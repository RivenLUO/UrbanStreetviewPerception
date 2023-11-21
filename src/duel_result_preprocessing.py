"""
     Functions for preprocessing data that used for the project.
"""

import pandas as pd
from keras.utils import load_img, img_to_array


def load_csv_to_df(duel_results_path, keep_nopref=False):
    """
    Load duel results from csv file.

    Args:
        duel_results_path (str):
            Path to the csv file containing duel results.
        keep_nopref (bool):
            Whether to keep the no preference data.

    Returns:
        A pandas.DataFrame instance containing duel results.
    """
    # Define columns
    duel_results = pd.read_csv(duel_results_path, usecols=[0, 1, 2], header=None)
    duel_results.columns = ['IMG1', 'IMG2', 'LABEL']

    # Deleting the no preference data
    if not keep_nopref:
        duel_results = duel_results[duel_results['LABEL'] != "No preference"]
        duel_results.reset_index(drop=True, inplace=True)

    return duel_results


def remove_duplicates(df):

    """
    Remove the duels that contain the same image pairs. A 'IMG_PAIR' column will be added to the dataframe.

    For example, if the duel results contain the following duels:
                    IMG1    IMG2    LABEL
                    1       2       left
                    1       2       left
                    2       1       right
                    3       4       left
                    4       3       right
                    5       6       left
                    6       5       left
                    7       8       left
                    7       8       right

                    The function will remove the duels with the same image pairs
                    and keep the first one,
                    i.e., The remaining duels will be:
                    IMG1    IMG2    LABEL
                    1       2       left
                    3       4       left

    Args:
        df (pd.DataFrame): The duel results in a dataframe to be processed.

    Returns:
        (pd.DataFrame): The duel results without the duplicated duels
    """

    # Print the number of duels before removing the duplicated duels
    print(f"Number of duels before removing the duplicated duels: {len(df)}")

    # Remove the duplicated rows that have the same image pair and label
    df.drop_duplicates(subset=['IMG1', 'IMG2', 'LABEL'], inplace=True, keep='first')

    # Remove all the duplicated rows that have the same image pair but different label
    df['IMG_PAIR'] = df.apply(lambda x: tuple([x['IMG1'], x['IMG2']]), axis=1)
    img_pair_indices = []
    for img_pair in df['IMG_PAIR'].value_counts().index:
        if df[df['IMG_PAIR'] == img_pair]['LABEL'].nunique() > 1:
            img_pair_indices.append(img_pair)
            df.drop(df[df['IMG_PAIR'] == img_pair].index, inplace=True)

    # Remove the duplicated rows that have the reversed image pair in img_pair_indices
    df['IMG_PAIR_Reversed'] = df.apply(lambda x: tuple(reversed(x['IMG_PAIR'])), axis=1)
    df.drop(df[df['IMG_PAIR_Reversed'].isin(img_pair_indices)].index, inplace=True)

    # Remove the duplicated rows that like (IMG1, IMG2, LABEL) and (IMG2, IMG1, LABEL)
    df['IMG_PAIR_Unique'] = df.apply(lambda x: tuple(sorted([x['IMG1'], x['IMG2']])), axis=1)
    # If the labels are the same, remove all the duplicated rows
    df.drop_duplicates(subset=['IMG_PAIR_Unique', 'LABEL'], inplace=True, keep=False)
    # If the labels are different, keep the first row
    df.drop_duplicates(subset=['IMG_PAIR_Unique'], inplace=True, keep='first')

    # Remove the columns that are no longer needed
    df.drop(columns=['IMG_PAIR_Reversed', 'IMG_PAIR_Unique'], inplace=True)

    # Print the number of duels after removing the duplicated duels
    print(f"Number of duels after removing the duplicated duels: {len(df)}")

    return df


def format_label(label):
    if label == 'left':
        return 1.0
    elif label == 'right':
        return 0.0
    else:
        raise ValueError(f"Invalid duel result label: {label}. Expected 'left' or 'right'.")


def preprocess_image(file_path, img_size):
    img = load_img(file_path, target_size=(img_size, img_size), color_mode='rgb', interpolation='bicubic')
    img = img_to_array(img)
    return img


def preprocess_duel(img_size, image1_path, image2_path, duel_results):
    image_pair = preprocess_image(image1_path, img_size), preprocess_image(image2_path, img_size)
    label = format_label(duel_results)
    return image_pair, label
