import os
import time
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

from agents.common import PLAYER1, PLAYER2


def get_mlp() -> MLPClassifier:
    """
    Return a standard configuration for the MLP.
    Hyper-Parameters found by Grid Search.

    Returns
    -------
    A Multilayer Perceptron instance
    """
    return MLPClassifier(activation='logistic',
                         hidden_layer_sizes=(126 * 6),
                         max_iter=1500,
                         alpha=0.001,
                         n_iter_no_change=10,
                         learning_rate='adaptive',
                         learning_rate_init=0.01,
                         shuffle=True,
                         tol=0.0001,
                         # warm_start=True,
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

    # unique boards and inverse index mapping
    X_new, idx_inv = np.unique(X, axis=0, return_inverse=True)

    # init new y
    y_new = np.zeros((X_new.shape[0],), dtype=np.int8)
    for i in range(y_new.shape[0]):
        # moves corresponding to unique boards
        y_to_unique_entry = y[idx_inv == i]
        # frequency of moves
        columns, counts = np.unique(y_to_unique_entry, return_counts=True)
        # most frequent move
        y_new[i] = columns[np.argmax(counts)].astype(np.int8)

    return X_new, y_new


def insert_flipped_boards(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inserts flipped boards and moves to dataset.

    Parameters
    ----------
    X : np.ndarray
        Data in shape(n_samples, n_features)
    y : np.ndarray
        Labels in shape(n_samples, )

    Returns
    -------
    Extended X and y
    """

    def flip_move(move):
        return move - 2 * (move - 3)

    def flip_board(board):
        return np.flip(board.reshape(board.shape[0], 6, 7), axis=2).flatten()

    y_flipped = flip_move(y)
    X_flipped = flip_board(X).reshape(X.shape[0], 42)

    return np.concatenate((X, X_flipped), axis=0), np.concatenate((y, y_flipped), axis=0)


def split_predicted(X: np.ndarray, y: np.ndarray, mlp: MLPClassifier) -> Tuple[np.ndarray, np.ndarray,
                                                                               np.ndarray, np.ndarray]:
    y_hat = mlp.predict(X)

    X_pred, y_pred = X[y_hat == y], y[y_hat == y]
    X_npred, y_npred = X[y_hat != y], y[y_hat != y]

    return X_pred, X_npred, y_pred, y_npred


if __name__ == '__main__':

    # starting from scratch maybe

    INIT_MODEL = ''  # 'models/40_WARM_50_200.pkl'  # 'models/6_10000_10_backup.pkl'
    MODEL_PATH = 'ITER_30_500.pkl'
    INIT_DATASET = ''  # 'data/7_10000_10_init.mat'  # 'data/26_MLP_RA_10000_50_05_init.mat'
    DATASET = 'ITER_30_500.mat'
    DATA_FOLDER = 'data/'
    MODELS_FOLDER = 'models/'
    N_MATCHES = 50
    N_ITER = 10
    SA_RATIO = -1

    # iteratively increase dataset by new game results
    # save dataset after every iteration


    X, y = None, None


    t00 = time.time()

    # create folders if not already existing
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if not os.path.exists(MODELS_FOLDER):
        os.mkdir(MODELS_FOLDER)

    # if init model is not given, create first model by training on random vs random dataset
    if INIT_DATASET == '' or INIT_DATASET is None:
        # generate init data
        print('Generating init Data...')
        t0 = time.time()
        boards, moves, _ = auto_rematch(random_move, random_move, args_1=tuple({True}), args_2=tuple({True}),
                                        n_matches=N_MATCHES)
        save_data(boards, moves, filename=DATA_FOLDER + '0_' + DATASET)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

        # prepare data
        print('Preparing Data...')
        t0 = time.time()
        data = scipy.io.loadmat(DATA_FOLDER + '0_' + DATASET)
        X, y = data['data'], data['labels'][0, :]
    else:
        # load init dataset
        print(f'Preparing Data {INIT_DATASET}...')
        t0 = time.time()
        data = scipy.io.loadmat(INIT_DATASET)
        X, y = data['data'], data['labels'][0, :]

    # insert flipped boards, clean, encode, split
    X, y = insert_flipped_boards(X, y)
    X, y = clean_data(X, y)
    scipy.io.savemat(DATA_FOLDER + '0_GROWING.mat', {'data': X, 'labels': y})
    X = OneHotEncoder(categories=[[-1, 0, 1]] * 42).fit_transform(X).toarray().astype(np.int8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    td = time.time()
    print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

    # train on data
    print('Training MLP...')
    print(f'Training on {len(y_train)} entries...')
    t0 = time.time()
    mlp = get_mlp()
    mlp.fit(X_train, y_train)
    td = time.time()
    print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')
    print('MLP score (Train):', mlp.score(X_train, y_train))
    print('MLP score (Test): ', mlp.score(X_test, y_test))

    # save model
    joblib.dump(mlp, MODELS_FOLDER + '0_' + MODEL_PATH)






    # start iterative process
    for i in range(N_ITER):
        print('-' * 50)
        print('\nITERATION', i+1)
        # load new mlp
        mlp = joblib.load(MODELS_FOLDER + str(i) + '_' + MODEL_PATH)

        # generate data (random vs mlp)
        print('Generating Data...')
        t0 = time.time()
        boards, moves, a_wins = auto_rematch(mlp_move, random_move, args_1=tuple({mlp}), args_2=tuple({True}),
                                             n_matches=int(N_MATCHES), sa_ratio=SA_RATIO)
        print(np.float32(a_wins[PLAYER1] / (2 * N_MATCHES)).round(5))
        save_data(boards, moves, filename=DATA_FOLDER + str(i+1) + '_' + DATASET)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

        # prepare data
        print('Preparing Data...')
        t0 = time.time()
        data = scipy.io.loadmat(DATA_FOLDER + str(i+1) + '_' + DATASET)
        X_new, y_new = data['data'], data['labels'][0, :]

        data_grow = scipy.io.loadmat(DATA_FOLDER + str(i) + '_GROWING.mat')
        X, y = data_grow['data'], data_grow['labels'][0, :]

        # insert flipped boards, clean, encode, split
        X, y = insert_flipped_boards(X, y)
        X, y = clean_data(X, y)
        if X is None or y is None:
            X, y = X_new, y_new
        else:
            X, y = np.concatenate((X, X_new), axis=0), np.concatenate((y, y_new), axis=0)
        scipy.io.savemat(DATA_FOLDER + str(i+1) + '_GROWING.mat', {'data': X, 'labels': y})
        print(f'{len(y)} entries in total')
        X = OneHotEncoder(categories=[[-1, 0, 1]] * 42).fit_transform(X).toarray().astype(np.int8)

        # new entries
        # _, Xnp, _, ynp = split_predicted(X, y, mlp)
        # npred_pred_ratio = np.minimum(len(ynp) / len(yp), 0.3)
        # substitute some predicted boards by non-predicted boards
        # X, _, y, _ = train_test_split(Xp, yp, test_size=npred_pred_ratio, stratify=yp)
        # X, y = np.concatenate((X, Xnp), axis=0), np.concatenate((y, ynp), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

        # train on data
        print('Training MLP...')
        print(f'Training on {len(y_train)} entries...')
        t0 = time.time()
        # mlp = get_mlp()
        mlp.fit(X_train, y_train)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')
        print('MLP score (Train):', mlp.score(X_train, y_train))
        print('MLP score (Test): ', mlp.score(X_test, y_test))

        # save model
        joblib.dump(mlp, MODELS_FOLDER + str(i+1) + '_' + MODEL_PATH)

    tdd = time.time()
    print(f'\nFinished after {((tdd - t00) // 3600):.0f} h {((tdd - t00) // 60):.0f} min {((tdd - t00) % 60):.1f} sec')
