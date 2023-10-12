"""
    This module contains some useful functions for the project.
"""

import json
import csv
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np


def safe_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        print(f"Directory {dir} created!")
    else:
        raise ValueError(f"Directory {dir} already exists!")


def timeit(function):
    """
    Decorator who prints the execution time of a function

    :param function: function to be executed
    :type function: function

    :return:
    :rtype: function
    """

    def timed(*args, **kw):
        ts = time.time()
        print('\nExecuting %r ' % function.__name__)
        result = function(*args, **kw)
        te = time.time()
        print('\n%r executed in %2.2f s' % (function.__name__, (te - ts)))
        return result

    return timed


def plot_model_metrics(history, metrics, save_dir=None):
    """
    Plot the metrics of a model.

    :param history: model history
    :type history: keras.callbacks.History
    :param metrics: list of metrics to plot
    :type metrics: list
    :param save_dir: path of the directory where to save the figure
    :type save_dir: str

    :return: print a message if the figure is saved
    :rtype: str
    """

    # Check if the metric type is valid and raise an error if not in keras history
    for metric in metrics:
        if metric not in history.history.keys():
            raise ValueError(f'The metric {metric} is not in the keras history! Check the metric type.')

    # Plot the metrics by metric types
    for metric in metrics:
        plt.figure()
        epochs = np.arange(1, len(history.history[metric]) + 1, dtype=int)
        plt.plot(epochs, history.history[metric], 'b', label='Train ' + metric.title())
        plt.plot(epochs, history.history['val_' + metric], 'r', label='Validation ' + metric.title())
        plot_title = 'Training and Validation' + metric.title()
        plt.title(plot_title)
        plt.ylabel(metric.title())
        plt.xlabel('Epoch')
        # plt.xticks(range(len(history.history[metric])))
        plt.legend()

        # Save the figure if a path is provided
        if save_dir is not None:
            if os.path.exists(os.path.join(save_dir, plot_title + ".png")):
                return print("The figure already exists! Check the save directory.")

            else:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, plot_title + ".png"))
                print(f'Figure {plot_title} saved!')

        # Show the figure
        plt.show()
        plt.close()


def safe_save_model_history(history, save_dir, save_type):
    """
    Save the model history in a specified file.

    :param history: model history
    :type history: keras.callbacks.History
    :param save_dir: path of the directory where to save the file
    :type save_dir: str
    :param save_type: type of the file to save
    :type save_type: str

    :return: print a message if the file is saved
    :rtype: str
    """

    # Save model history avoiding overwriting
    if os.path.exists(os.path.join(save_dir, "model_history.txt")):
        return print("The model history file already exists! Check the save directory.")

    else:
        # Save the history in a txt file if specified:
        if save_type == "txt":
            with open(os.path.join(save_dir, "history.txt"), "w") as f:
                f.write(str(history.history))
                f.close()
            return print("Model history saved to txt!")

        # Save the history in a json file if specified:
        elif save_type == "json":
            with open(os.path.join(save_dir, "history.json"), "w") as f:
                json.dump(history.history, f, indent=4)
                f.close()
            return print("Model history saved to json!")

        # Save the history in a csv file if specified:
        elif save_type == "csv":
            with open(os.path.join(save_dir, "history.csv"), "w", newline='') as f:
                # Create the csv writer
                writer = csv.writer(f)
                # Write the header
                writer.writerow(history.history.keys())
                # Write the values
                for row in zip(*history.history.values()):
                    writer.writerow(row)
            return print("Model history saved to csv!")

        # Save the history in a pickle file if specified:
        elif save_type == "pickle":
            with open(os.path.join(save_dir, "history.pickle"), "wb") as f:
                pickle.dump(history.history, f)
            return print("Model history saved to pickle!")

        # Raise an error if the save type is not supported:
        else:
            raise ValueError('The save type is not supported! Check the save type. Supported types: txt, json, csv, '
                             'pickle.')


def safe_save_model_architecture(model, save_dir):
    """
    Save the model architecture in a json file.

    :param model: keras model
    :type model: keras.Model
    :param save_dir: path of the directory where to save the file
    :type save_dir: str

    :return: print a message if the file is saved
    :rtype: str
    """

    # Save model architecture avoiding overwriting
    if os.path.exists(os.path.join(save_dir, "model_architecture.json")):
        return print("The json file already exists! Check the save directory.")

    else:
        with open(os.path.join(save_dir, f"{model.name}_architecture.json"), "w") as f:
            f.write(model.to_json(indent=4))
            f.close()
        return print("Model architecture saved!")


def safe_save_model_weights(model, save_dir):
    """
    Save the model weights in a h5 file.

    :param model: keras model
    :type: keras.Model
    :param save_dir: path of the directory where to save the file
    :type: str

    :return: print a message if the file is saved
    :rtype: str
    """
    # Save model architecture avoiding overwriting
    if os.path.exists(os.path.join(save_dir, "model_weights.h5")):
        return print("The weights file already exists! Check the save directory.")

    else:
        model.save_weights(os.path.join(save_dir, f"{model.name}_weights.h5"))
        return print("Model weights saved!")


def safe_save_model(model, save_dir, model_save_type):
    """
    Save the model in a specified file.

    :param model_save_type:
    :param model: keras model
    :type: keras.Model
    :param save_dir: path of the directory where to save the file
    :type: str

    :return: print a message if the file is saved
    :rtype: str
    """
    # Save model architecture avoiding overwriting
    if os.path.exists(os.path.join(save_dir, "model.h5")):
        return print("The model file already exists! Check the save directory.")

    else:
        if model_save_type == 'h5':
            model.save(os.path.join(save_dir, f"{model.name}.h5"))
            return print("Model saved to h5!")

        elif model_save_type == 'keras':
            model.save(os.path.join(save_dir, f"{model.name}.keras"))
            return print("Model saved to keras!")

        else:
            raise ValueError('The save type is not supported! Check the save type. Supported types: h5, keras.')


def safe_save_hyperparams(hyperparams_dict, save_dir):
    """
    Save the hyperparams in a json file.

    :param hyperparams_dict:
    :param save_dir:

    :return:
    :rtype:
    """
    # Save model architecture avoiding overwriting
    if os.path.exists(os.path.join(save_dir, "hyperparams.json")):
        return print("The hyperparams file already exists! Check the save directory.")

    else:
        with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
            json.dump(hyperparams_dict, f, indent=4)
            f.close()
        return print("Hyperparams saved!")


def safe_save_training_results(hyperparams_dict, model, model_save_type, history, his_save_type, save_dir):
    """
    Save the training results including all in a directory.

    :param his_save_type:
    :param model_save_type:
    :param history:
    :param hyperparams_dict:
    :type hyperparams_dict:
    :param model:
    :type model:
    :param save_dir:
    :type save_dir:

    :return:
    :rtype:
    """
    if os.path.exists(save_dir):
        raise ValueError("The save directory already exists! Check the save directory.")


    else:
        os.makedirs(save_dir, exist_ok=True)

        if hyperparams_dict is not None:
            safe_save_hyperparams(hyperparams_dict, save_dir)

        # Save the model history
        safe_save_model_history(history, save_dir, his_save_type)

        # Save the model architecture
        safe_save_model_architecture(model, save_dir)

        # Save the model weights
        safe_save_model_weights(model, save_dir)

        # Save the model
        safe_save_model(model, save_dir, model_save_type)


def safe_save_Hannick_training_results(model, model_save_type, history, his_save_type, save_dir):
    if os.path.exists(save_dir):
        raise ValueError("The save directory already exists! Check the save directory.")


    else:
        os.makedirs(save_dir, exist_ok=True)

        # Save the model history
        safe_save_model_history(history, save_dir, his_save_type)

        # Save the model architecture
        safe_save_model_architecture(model, save_dir)

        # Save the model weights
        safe_save_model_weights(model, save_dir)

        # Save the model
        safe_save_model(model, save_dir, model_save_type)
