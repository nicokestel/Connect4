import numpy as np
import scipy.io
from sys import getsizeof

if __name__ == '__main__':
    boards = list()
    labels = list()
    for l in range(1000000):
        board = (np.random.rand(6, 7) * 5).flatten()
        boards.append(board.astype(np.int8))

        label = l % 7
        labels.append(np.int8(label))

    scipy.io.savemat('data.mat', {'data': boards, 'labels': labels})

    dataset = scipy.io.loadmat('data.mat')
    data = dataset['data']
    labels = dataset['labels'][0, :]

    print(type(data), type(data[0]), type(data[0][0]))
    print(type(labels), type(labels[0]))

    print(data.shape, labels.shape)

    print(data[0], labels[0])