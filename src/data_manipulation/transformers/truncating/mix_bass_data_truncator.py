import numpy as np


def data_initial_truncator(data, min_dimension):
    # Keep track of original lengths for masking
    data["sequence_lengths"] = [arr.shape[1] for arr in data["x"]]

    data["x"] = [arr[:, :min_dimension] for arr in data["x"]]
    data["y"] = [arr[:, :min_dimension] for arr in data["y"]]

    return data


def data_overall_truncator(data, min_dimension_train, min_dimension_test=np.inf):
    min_dimension = min(min_dimension_train, min_dimension_test)

    data["x"] = [arr[:, :min_dimension] for arr in data["x"]]
    data["y"] = [arr[:, :min_dimension] for arr in data["y"]]

    return data
