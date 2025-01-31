import numpy as np


def data_padder(data, max_dimension_train, max_dimension_test=0):
    # Keep track of original lengths for masking
    data["sequence_lengths"] = [arr.shape[1] for arr in data["x"]]

    max_dimension = max(max_dimension_train, max_dimension_test)
    data["x"] = [
        np.pad(arr, ((0, 0), (0, max_dimension - arr.shape[1]))) for arr in data["x"]
    ]
    data["y"] = [
        np.pad(arr, ((0, 0), (0, max_dimension - arr.shape[1]))) for arr in data["y"]
    ]
    return data
