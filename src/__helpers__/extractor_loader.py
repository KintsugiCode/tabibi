import numpy as np
from numpy.lib.npyio import NpzFile


def extract_numpy_data(file_path):
    """
    Extract data from a numpy file.

    Args:
        file_path (str): Path to the numpy file.

    Returns:
        dict or numpy.ndarray: Extracted data.
    """
    data = np.load(f"{file_path}", allow_pickle=True)

    if isinstance(data, np.recarray):
        extracted_data = {item['key']: item['value'] for item in data}
    elif isinstance(data, np.ndarray):
        extracted_data = np.array(data)
    elif isinstance(data, NpzFile):
        extracted_data = np.array([item for item in data["arr_0"]])
        print(extracted_data)
    else:
        raise Exception("The transformed data saved has to be in a format of numpy array or dict of arrays")

    return extracted_data
