import os
import re
import pandas as pd
import numpy as np
from keras.utils import img_to_array, load_img, save_img
import tensorflow as tf


class ImageArrayProcessor:
    pass


class DuelResultsProcessor:

    def __init__(self, duel_results_path, remove_no_preference=True):
        self.duel_results_path = duel_results_path
        self.duel_results = pd.read_csv(duel_results_path, usecols=[0, 1, 2], header=None)
        self.duel_results.columns = ['IMG_LEFT', 'IMG_RIGHT', 'LABEL']
        self.duel_question_idx = int(duel_results_path.split('\\')[-1].split('_')[-1].split('.')[0])

        if remove_no_preference:
            self.remove_no_preference()

    def create_data_loading_mapping(self, image_dir, save=False, save_path=None):
        """
        """
        mapping_df = pd.DataFrame(columns=['IMG_LEFT', 'IMG_RIGHT', 'LABEL'])

        image_names = pd.DataFrame(os.listdir(image_dir), columns=['IMG_NAME'])
        image_names = image_names[image_names['IMG_NAME'].str.endswith('.jpg')]
        image_names_set = set(image_names['IMG_NAME'])

        for duel_idx, duel_row in self.duel_results.iterrows():

            left_image_match = [image_name for image_name in image_names_set if duel_row['IMG_LEFT'] in image_name]
            right_image_match = [image_name for image_name in image_names_set if duel_row['IMG_RIGHT'] in image_name]

            if left_image_match:
                mapping_df.loc[duel_idx, 'IMG_LEFT'] = os.path.join(image_dir, left_image_match[0])

            if right_image_match:
                mapping_df.loc[duel_idx, 'IMG_RIGHT'] = os.path.join(image_dir, right_image_match[0])

            mapping_df.loc[duel_idx, 'LABEL'] = duel_row['LABEL']

        print(
            f"Total number of duels: {len(mapping_df)}\n"
        )

        if save is True:
            if save_path is not None:
                save_path = os.path.join(
                    os.path.dirname(self.duel_results_path),
                    f'duel_question_{self.duel_question_idx}_-data_loading_index.csv'
                )
                mapping_df.to_csv(save_path, index=False)
                print(f"Duel results of question {self.duel_question_idx} mapping file saved to {save_path}")

            else:
                raise ValueError("Please provide a save path.")

        return mapping_df

    def get_labels(self, formatted=None):
        """
        """
        labels = self.duel_results['LABEL']
        if formatted:
            label_formatted = []

            if formatted == "comparison":

                for label in labels:

                    if label == "left":
                        label_formatted.append([int(1), int(0)])

                    elif label == "right":
                        label_formatted.append([int(0), int(1)])

                    else:
                        label_formatted.append([int(-1), int(-1)])

                return label_formatted

            elif formatted == "ranking":

                for label in labels:

                    if label == "left":
                        label_formatted.append(int(1))

                    elif label == "right":
                        label_formatted.append(int(0))

                    else:
                        label_formatted.append(int(-1))

                return label_formatted

            else:
                raise ValueError("Invalid model type. Please choose between 'comparison' and 'ranking'.")

        else:
            return labels

    def remove_no_preference(self):
        """
        """
        self.duel_results = self.duel_results[self.duel_results['LABEL'] != "no_preference"]


class DatasetProcessor:

    def __init__(self, model_type, data_dir, duel_results_path, dataset=None):

        self.model_type = model_type
        self.data_dir = data_dir
        self.duel_results_path = duel_results_path

        self.duel_results_processor = DuelResultsProcessor(duel_results_path)
        self.data_loading_mapping = self.duel_results_processor.create_data_loading_mapping(self.data_dir)
        self.labels = self.duel_results_processor.get_labels(formatted=self.model_type)

        if dataset is not None:
            self.validate_and_set_dataset(dataset)
            self.dataset = dataset

        else:
            self.dataset = self.load_data()

    def validate_and_set_dataset(self, dataset):
        if self.model_type == "comparison":
            if len(dataset) != 3:
                raise ValueError("Invalid dataset. Provide a list in the form [image1_array, image2_array, labels].")
            if dataset[0].shape != dataset[1].shape:
                raise ValueError("Invalid dataset. Image arrays must have the same shape.")
            if dataset[2].shape[1] != 2:
                raise ValueError("Invalid dataset. Labels array should have shape (n, 2).")

        elif self.model_type == "ranking":
            if len(dataset) != 3:
                raise ValueError("Invalid dataset. Provide a list in the form [image1_array, image2_array, labels].")
            if dataset[0].shape != dataset[1].shape:
                raise ValueError("Invalid dataset. Image arrays must have the same shape.")
            if dataset[2].shape[1] != 1:
                raise ValueError("Invalid dataset. Labels array should have shape (n,).")

    def load_data(self, left_image_array=None, right_image_array=None, labels=None):

        if left_image_array is not None and right_image_array is not None and labels is not None:
            self.validate_and_set_dataset([left_image_array, right_image_array, labels])
            self.dataset = [left_image_array, right_image_array, labels]
        else:
            raise ValueError(
                f"One or more of the following are not provided: left_image_array, right_image_array, labels. "
                f"Since you are loading data by providing the arrays, please provide all three."
            )

        if self.data_loading_mapping is not None:

            left_image_array = []
            right_image_array = []

            for duel_idx, duel_row in self.data_loading_mapping.iterrows():
                left_image_array.append(img_to_array(load_img(duel_row['IMG_LEFT'])))
                right_image_array.append(img_to_array(load_img(duel_row['IMG_RIGHT'])))

            left_image_array = np.array(left_image_array)
            right_image_array = np.array(right_image_array)

            self.dataset = [left_image_array, right_image_array, self.labels]

        else:
            raise ValueError(
                f"Neither the mapping file nor the data arrays and labels are provided. "
                f"Please provide the mapping file or the data arrays and labels."
            )

    def split_dataset(self, split_ratio=None):
        if split_ratio is None:
            split_ratio = [0.6, 0.2]

        total_samples = len(self.dataset[0])
        train_size = int(split_ratio[0] * total_samples)
        valid_size = int(split_ratio[1] * total_samples)

        X_train = [self.dataset[0][:train_size], self.dataset[1][:train_size]]
        y_train = self.dataset[2][:train_size]

        X_valid = [self.dataset[0][train_size:train_size + valid_size],
                   self.dataset[1][train_size:train_size + valid_size]]
        y_valid = self.dataset[2][train_size:train_size + valid_size]

        X_test = [self.dataset[0][train_size + valid_size:], self.dataset[1][train_size + valid_size:]]
        y_test = self.dataset[2][train_size + valid_size:]

        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)





class StreetImage:
    def __init__(self, **kwargs):
        self.image = None
        self.identifier = None
        self.coordinates = None
        self.time_stamp = None
        self.image_file_name = None
        self.file_name_pattern = r'(-?\d+\.\d+)_(-?\d+\.\d+)__(\w+?)__(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.(\w+)'

    def load_image(self, file_path, **kwargs):
        self.image = img_to_array(load_img(file_path, **kwargs))
        self.image_file_name = os.path.basename(file_path)
        self.get_info_from_file_name()

    def save_image(self, file_path, **kwargs):
        save_img(file_path, self.image, **kwargs)

    def get_info_from_file_name(self):
        match = re.match(self.file_name_pattern, self.image_file_name)
        if match:
            self.coordinates = (float(match.group(1)), float(match.group(2)))
            self.identifier = match.group(3)
            self.time_stamp = match.group(4)
        else:
            raise ValueError('No match found. Invalid file name or pattern.')


class StreetImageDuel:
    def __init__(self, image1=None, image2=None, label=None):
        self.image1 = image1
        self.image2 = image2
        self.label = label
        self.image_pair = (self.image1, self.image2)
        self.duel = [[self.image1, self.image2], self.label]
        self.duel_title = None
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.duel):
            duel = self.duel[self.index]
            self.index += 1
            return duel
        else:
            raise StopIteration

    def set_duel(self, image1, image2, label):
        self.image1 = image1
        self.image2 = image2
        self.label = label

    def get_image_pair(self):
        return self.image_pair

    def get_label(self):
        return self.label

    def format_label(self, model_type):
        if model_type == 'comparison':
            if self.label == 'left':
                return [1, 0]
            elif self.label == 'right':
                return [0, 1]
            else:
                return [0, 0]
        elif model_type == 'ranking':
            if self.label == 'left':
                return 1
            elif self.label == 'right':
                return 0
            else:
                return -1


def format_label(label, model_type):
    if model_type == 'comparison':
        if label == 'left':
            return [1, 0]
        elif label == 'right':
            return [0, 1]
        else:
            return [0, 0]
    elif model_type == 'ranking':
        if label == 'left':
            return 1
        elif label == 'right':
            return 0
        else:
            return -1


class StreetImageDuelSet:
    """
    ImageDuelSet is a collection of ImageDuel objects.
    Use Pandas DataFrame to store the duels to facilitate data manipulation.
    """

    def __init__(self, duels: list = None, duel_title: str = None):
        new_duels = []
        for duel in duels:
            # convert to ImageDuel object
            if not isinstance(duel, StreetImageDuel):
                duel = StreetImageDuel(duel[0][0], duel[0][1], duel[1])
                new_duels.append(duel)
        self.duels = new_duels
        self.duel_title = duel_title
        self.num_duels = len(self.duels)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.num_duels:
            result = self.duels[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def _validate_duels(self, duels):
        # type check
        type_flag = [isinstance(duel, StreetImageDuel) for duel in duels]

        if not isinstance(duels, list) and not all(type_flag):
            raise ValueError('self.duels must be a list of ImageDuel objects')
        print(f'{duels} is a list of ImageDuel objects')

        # title check
        title_flag = [duel.duel_title == self.duel_title for duel in duels]

        if not all(title_flag):
            raise ValueError('All duels must have the same duel_title')
        print(f'{duels} matches self.duel_title:{self.duel_title}')

    def concatenate_duels(self, duels):
        self._validate_duels(duels)
        self.duels += duels

    def get_duel_slice(self, start, end):
        return self.duels[start:end]

    def check_duel_conflict(self, return_flag=False):
        """
        Check if there is any conflict in the duel set.
        :return: a list of conflicting duels
        """
        duel_groups = {}  # Dictionary to group duels by their image pairs
        conflict_duels = []

        for duel in self.duels:
            image_pair = tuple(sorted(duel.image_pair))  # Ensure consistent order
            if image_pair not in duel_groups:
                duel_groups[image_pair] = [duel]
            else:
                duel_groups[image_pair].append(duel)

        for group in duel_groups.values():
            if len(group) > 1:
                # Check for conflicts within the group
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        if group[i].label != group[j].label:
                            conflict_duels.extend([group[i], group[j]])

        if not conflict_duels:
            print('No conflict duels found.')

        else:
            print(f'Conflict duels found: {conflict_duels}')
            if return_flag:
                return conflict_duels

    def remove_duplicate(self):
        """
        Remove duplicate duels from the dataset.
        For example, if the dataset contains [[image1, image2], 1] and [[image1, image2], 1], the latter will be removed.
        or if the dataset contains [[image1, image2], 1] and [[image2, image1], 2], the latter will be removed.
        :return:
        """
        unique_duels = []
        seen_pairs = set()

        for duel in self.duels:
            image_pair = tuple(sorted(duel.image_pair))  # Ensure consistent order
            if image_pair not in seen_pairs:
                unique_duels.append(duel)
                seen_pairs.add(image_pair)

        return unique_duels

    def prepare_tf_dataset_for_siamese_network(self, model_type):
        """
        Prepare the dataset for siamese network training.
        """

        def preprocess_duel(duel, model_type):
            image_pair, label = duel
            # Convert NumPy arrays to TensorFlow tensors
            image_pair = (tf.convert_to_tensor(image_pair[0], dtype=tf.float32),
                          tf.convert_to_tensor(image_pair[1], dtype=tf.float32))
            # format label
            label = tf.convert_to_tensor(format_label(label,model_type), dtype=tf.float32)
            return image_pair, label

        processed_data = [preprocess_duel(duel,model_type) for duel in self.duels]

        return tf.data.Dataset.from_tensor_slices(processed_data)
