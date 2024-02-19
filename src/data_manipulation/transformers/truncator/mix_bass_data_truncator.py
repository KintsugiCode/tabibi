import numpy as np


def data_truncator(
    data,
    flag,
    min_dimension=np.inf,
    min_dimension_train=np.inf,
    min_dimension_test=np.inf,
):
    """
    Initial truncation is for use within one dataset.

    Overall truncation is for use over multiple datasets.

    Initial truncation, before knowing all datapoint dimensions over all datasets, is necessary to be able to convert
    datasets into numpy recarrays and therefore already needs to occur during individual dataset pre-processing.
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
