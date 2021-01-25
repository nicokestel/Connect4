import numpy as np
import scipy.io
import os


def test_save_data():
    from agents.agent_mlp.mlp_training.generate_data import save_data

    N = 10
    filename = 'data-test.mat'

    boards = list()
    labels = list()

    for l in range(N):
        board = (np.random.rand(6, 7) * 5).flatten()
        boards.append(board.astype(np.int8))

        label = l % 7
        labels.append(np.int8(label))

    entries = save_data(data=boards, labels=labels, filename=filename)

    assert entries == np.int64(N)

    dataset = scipy.io.loadmat(filename)
    os.remove(filename)

    data = dataset['data']
    labels = dataset['labels'][0, :]

    assert type(data) == np.ndarray
    assert data.shape == (N, 42)
    assert type(data[0]) == np.ndarray
    assert data[0].shape == (42,)
    assert type(data[0][0]) == np.int8

    assert type(labels) == np.ndarray
    assert labels.shape == (N,)
    assert type(labels[0]) == np.int8
