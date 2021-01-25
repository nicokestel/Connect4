from typing import List
import numpy as np
import scipy.io


def save_data(data: List[np.ndarray], labels: List[np.int8], filename: str = 'C4-mlp-data.mat') -> np.int64:
    """

    Parameters
    ----------
    data : List[np.ndarray]
        Data to save
    labels : List[np.int8]
        Labels to save
    filename: str
        Filename

    Returns
    -------
    entries: np.int64
        Number of entries
    """
    scipy.io.savemat(filename, {'data': data, 'labels': labels})
    return len(labels)
