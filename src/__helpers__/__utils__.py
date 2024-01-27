import numpy as np
from numpy.lib.npyio import savez_compressed
import os


def load_numpy_data(file_path):
    data = np.load(f"{file_path}", allow_pickle=True)

    return {item["key"]: item["value"] for item in data["arr_0"]}


def get_one_file_with_extension(directory_path, extension):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                return file
    return None


def savez_numpy_data(file_path, data):
    savez_compressed(file_path, data)


def convert_t_dict_key_to_numpy_arrays(dictionary, keys):
    for key in keys:
        dictionary[key] = np.array(dictionary[key])

    return dictionary


def convert_to_recarray(data_dict):
    return np.rec.array(
        [(k, v) for k, v in data_dict.items()],
        dtype=[("key", object), ("value", object)],
    )
