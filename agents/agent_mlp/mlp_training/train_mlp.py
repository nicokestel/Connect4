import os
from typing import Tuple

import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from agents.agent_mlp.mlp_training.auto_rematch import auto_rematch
from agents.agent_mlp.mlp_training.generate_data import save_data
from agents.agent_random import generate_move as random_move
from agents.agent_mlp import generate_move as mlp_move

import scipy.io
import numpy as np


def get_mlp() -> MLPClassifier:
    """
    Return a standard configuration for the MLP.
    Hyper-Parameters found by Grid Search.

    Returns
    -------
    A Multilayer Perceptron instance
    """
    return MLPClassifier(activation='logistic',
                         hidden_layer_sizes=(126 * 5),
                         max_iter=1500,
                         alpha=0.001,
                         n_iter_no_change=10,
                         learning_rate='adaptive',
                         learning_rate_init=0.01,
                         shuffle=True,
                         tol=0.0001,
                         # random_state=1,
                         verbose=0)


def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes duplicates from X and assigns most frequent label from y.

    Parameters
    ----------
    X : np.ndarray
        Data in shape(n_samples, n_features)
    y : np.ndarray
        Labels in shape(n_samples, )

    Returns
    -------
    Cleaned X and y
    """

    # join X and y
    D = np.concatenate((X, y[:, np.newaxis]), axis=1)

    i = 0
    while i < D.shape[0]:
        # indices of equal boards
        eq_mask = np.all(D[:, :-1] == D[i, :-1], axis=1)

        if eq_mask.sum() > 1:
            # labels of equal boards
            y_tmp = D[:, -1][eq_mask]

            # most frequently played column
            columns, counts = np.unique(y_tmp, return_counts=True)
            mp_col = columns[np.argmax(counts)].astype(np.int8)

            # set labels of equal boards to most played column
            D[eq_mask, -1] = mp_col

        i += 1

    # remove duplicates
    D = np.unique(D, axis=0)

    # split into X and y
    return D[:, :-1], D[:, -1]


if __name__ == '__main__':
    MODEL_PATH = 'c4_mlp_model.pkl'
    INIT_DATASET = 'data/10_MLP_RA_init.mat'
    DATASET = 'MLP_RA.mat'
    DATA_FOLDER = 'data/'
    MODELS_FOLDER = 'models/'
    N_MATCHES = 3000
    N_ITER = 4

    # create folders if not already existing
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if not os.path.exists(MODELS_FOLDER):
        os.mkdir(MODELS_FOLDER)

    print('Initialize...')

    if INIT_DATASET == '' or INIT_DATASET is None:
        # generate init data
        print('Generating init Data...')
        boards, moves = auto_rematch(random_move, random_move, args_1=tuple({True}), args_2=tuple({True}), n_matches=N_MATCHES)
        save_data(boards, moves, filename=DATA_FOLDER + '0_' + DATASET)

        # prepare data
        print('Preparing Data...')
        data = scipy.io.loadmat(DATA_FOLDER + '0_' + DATASET)
        X, y = data['data'], data['labels'][0, :]
    else:
        # load init dataset
        print(f'Preparing Data {INIT_DATASET}...')
        data = scipy.io.loadmat(INIT_DATASET)
        X, y = data['data'], data['labels'][0, :]

    # clean, encode, split
    X, y = clean_data(X, y)
    X = OneHotEncoder(categories=[[-1, 0, 1]] * 42).fit_transform(X).toarray().astype(np.int8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # train on data
    print('Training MLP...')
    mlp = get_mlp()
    mlp.fit(X_train, y_train)

    print('MLP score (Train):', mlp.score(X_train, y_train))
    print('MLP score (Test): ', mlp.score(X_test, y_test))

    # save model
    joblib.dump(mlp, MODELS_FOLDER + '0_' + MODEL_PATH)

    for i in range(N_ITER):
        print('\nITERATION', i+1)
        # load new mlp
        mlp = joblib.load(MODELS_FOLDER + str(i) + '_' + MODEL_PATH)

        # generate data (random vs mlp)
        print('Generating Data...')
        boards, moves = auto_rematch(mlp_move, random_move, args_1=tuple({mlp}), args_2=tuple({True}), n_matches=N_MATCHES + (i*100))
        save_data(boards, moves, filename=DATA_FOLDER + str(i+1) + '_' + DATASET)

        # prepare data
        print('Preparing Data...')
        data = scipy.io.loadmat(DATA_FOLDER + str(i+1) + '_' + DATASET)
        X, y = data['data'], data['labels'][0, :]

        # clean, encode, split
        X, y = clean_data(X, y)
        X = OneHotEncoder(categories=[[-1, 0, 1]] * 42).fit_transform(X).toarray().astype(np.int8)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

        # train on data
        print('Training MLP...')
        mlp = get_mlp()
        mlp.fit(X_train, y_train)

        print('MLP score (Train):', mlp.score(X_train, y_train))
        print('MLP score (Test): ', mlp.score(X_test, y_test))

        # save model
        joblib.dump(mlp, MODELS_FOLDER + str(i+1) + '_' + MODEL_PATH)

    print('Finished')
