import os
from tqdm import tqdm
import pandas as pd
from keras.utils import load_img, img_to_array


def preprocess_image(file_path, img_size, **kwargs):
    image = load_img(file_path, target_size=(img_size, img_size), **kwargs)
    image = img_to_array(image)
    return image


def preprocess_pair(image1, image2, img_size, **kwargs):
    return preprocess_image(image1, img_size, **kwargs), preprocess_image(image2, img_size, **kwargs)


def load_duel_results(duel_results_path):
    # Define columns
    duel_results = pd.read_csv(duel_results_path, usecols=[0, 1, 2], header=None)
    duel_results.columns = ['IMG1', 'IMG2', 'LABEL']

    # Deleting the no preference data
    duel_results = duel_results[duel_results['LABEL'] != "No preference"]
    duel_results.reset_index(drop=True, inplace=True)

    return duel_results


def format_duel_result_label(duel_result_label, model_type):
    if model_type == 'ranking':
        if duel_result_label == 'left':
            return 1
        elif duel_result_label == 'right':
            return 0
        else:
            raise ValueError(f"Invalid duel result label: {duel_result_label}. Expected 'left' or 'right'.")

    elif model_type == 'comparison':
        if duel_result_label == 'left':
            return 1, 0
        elif duel_result_label == 'right':
            return 0, 1
        else:
            raise ValueError(f"Invalid duel result label: {duel_result_label}. Expected 'left' or 'right'.")


def preprocess_duel(image1_path, image2_path, img_size, duel_results, model_type, **kwargs):
    duel_image_pair = preprocess_pair(image1_path, image2_path, img_size, **kwargs)
    duel_label = format_duel_result_label(duel_results, model_type)
    return duel_image_pair, duel_label

