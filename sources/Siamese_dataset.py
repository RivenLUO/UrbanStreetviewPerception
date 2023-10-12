import os
import numpy as np
from keras.utils import load_img, img_to_array
from sources_NOUSE.data_augmentation_ import data_aug_with_contrast


def prepare_datasets_for_siamese_network(image_dir, label_path, **kwargs):
    labels = np.load(label_path)

    # Format the labels (left: 1, right: 0, no preference: -1)
    numeric_labels = np.array(np.vectorize(format_label)(labels[:, 1]), dtype=np.int32)
    label_indices = np.array(labels[:, 0], dtype=np.int32)
    labels_formatted = np.column_stack((label_indices, numeric_labels))

    image1_array_with_serial_number = load_images_as_np_array(image_dir, **kwargs['image1_settings'])
    image2_array_with_serial_number = load_images_as_np_array(image_dir, **kwargs['image2_settings'])

    # Sort the arrays and labels by the indices (the first column)
    image1_array_with_serial_number = sorted(image1_array_with_serial_number, key=lambda x: x[0])
    image2_array_with_serial_number = sorted(image2_array_with_serial_number, key=lambda x: x[0])

    sorted_array_indices = np.argsort(labels_formatted[:, 0])
    labels_formatted = labels_formatted[sorted_array_indices]

    # Check if the serial numbers match
    for image1, image2, label in zip(image1_array_with_serial_number, image2_array_with_serial_number,
                                     labels_formatted):
        if image1[0] != image2[0] or image1[0] != label[0]:
            raise ValueError("The serial numbers do not match.")

    # Extract the image arrays from the list
    image1_array = [image1[1] for image1 in image1_array_with_serial_number]
    image2_array = [image2[1] for image2 in image2_array_with_serial_number]

    # Use the mask to filter arrays and labels if the label is "No preference" (i.e. -1 if formatted)
    mask = [label != -1 for label in labels_formatted[:, 1]]
    filtered_image1_array = np.array(image1_array)[mask]
    filtered_image2_array = np.array(image2_array)[mask]
    labels_formatted = labels_formatted[mask]

    labels = labels_formatted[:, 1]
    # Apply data augmentation techniques to the filtered images
    if kwargs.get('data_aug_RandomContrast', False):
        filtered_image1_array, filtered_image2_array, labels = data_aug_with_contrast(
            filtered_image1_array,
            filtered_image2_array,
            labels
        )

    return [filtered_image1_array, filtered_image2_array], labels


def load_images_as_np_array(image_dir, target_size=(224, 224), **kwargs):
    # Check if the kwargs are valid
    allowed_kwargs = {'if_image_name_contain', 'target_size', 'interpolation', 'if_return_serial_numbers'}
    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError('Keyword argument not understood:', kwarg, 'Allowed arguments:', allowed_kwargs)

    image_name_specified_flags = kwargs.get('if_image_name_contain', None)
    interpolation = kwargs.get('interpolation', 'bicubic')
    if_return_serial_numbers = kwargs.get('if_return_serial_numbers', False)

    image_arrays = []
    image_array_with_serial_numbers = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            image_name_specified = True  # Flag to check if the image matches all kwargs

            # Iterate through 'image_name_specified_flags' and check if the image name contains all the flags
            if image_name_specified_flags:
                for specified_flag in image_name_specified_flags:
                    if specified_flag not in image_name:
                        image_name_specified = False
                        break

            if image_name_specified:
                # Extract the current image index from the image name
                image_idx = int(image_name.split('.')[0].split('_')[2])

                image_path = os.path.join(image_dir, image_name)
                image = load_img(image_path, target_size=target_size, interpolation=interpolation)
                image_array = img_to_array(image, dtype=np.float32)

                if if_return_serial_numbers:
                    image_array_with_serial_numbers.append([image_idx, image_array])

                else:
                    image_arrays.append(image_array)

            else:
                # Skip the image if it does not match the specified flags
                continue

    if if_return_serial_numbers:
        return image_array_with_serial_numbers

    return np.array(image_arrays)


def format_label(label):
    if label == "left":
        return int(1)
    elif label == "right":
        return int(0)
    else:
        return int(-1)
