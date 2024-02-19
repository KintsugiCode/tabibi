import numpy as np


def data_truncator(
    data,
    flag,
    min_dimension=np.inf,
    min_dimension_train=np.inf,
    min_dimension_test=np.inf,
):
    """
    If passed the dimension of the shortest datapoint in a dataset, this truncates all datapoints in that dataset to
    that dimension and stores that value for potential later use with masking.
    If passed the dimensions of the shortest datapoints found within each of two datasets, this truncates all
    datapoints in both those datasets to that dimension.
    """
    if flag == "initial":
        # Keep track of original lengths for masking
        data["sequence_lengths"] = [arr.shape[1] for arr in data["x"]]

    if flag == "overall":
        min_dimension = min(min_dimension_train, min_dimension_test)

    data["x"] = [arr[:, :min_dimension] for arr in data["x"]]
    data["y"] = [arr[:, :min_dimension] for arr in data["y"]]

    data["x_phase"] = [arr[:, :min_dimension] for arr in data["x"]]
    data["y_phase"] = [arr[:, :min_dimension] for arr in data["y"]]

    return data
