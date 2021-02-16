import os
import time
from typing import Tuple

import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from agents.agent_mlp.mlp_training.auto_rematch import auto_rematch
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
                         hidden_layer_sizes=(126 * 4),
                         max_iter=1500,
                         alpha=0.001,
                         n_iter_no_change=10,
                         learning_rate='adaptive',
                         learning_rate_init=0.005,
                         shuffle=True,
                         tol=0.0001,
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


def extract_one_to_win(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts boards and final winning move of every game.

    Parameters
    ----------
    X : np.ndarray
        Boards in chronological order.
    y : np.ndarray
        Moves corresponding to boards.

    Returns
    -------
    Final board and move of every game in X.
    """

    # indices of first boards of each game
    idx_one_to_win = np.array(np.where(abs(X).sum(axis=1) <= 1))[0, :]
    # final boards and moves in previous game
    return X[idx_one_to_win - 1], y[idx_one_to_win - 1]


if __name__ == '__main__':

    INIT_MODEL = ''
    MODEL_PATH = 'INC_FIN.pkl'
    INIT_DATASET = 'data/BACKUP.mat'  # needs to be sorted in a chronological matter
    DATASET = 'INC_FIN.mat'
    DATA_FOLDER = 'data/'
    MODELS_FOLDER = 'models/'
    N_MATCHES = 5000
    N_ITER = 3
    SA_RATIO = -1.0

    X, y = None, None

    t00 = time.time()

    # create folders if not already existing
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if not os.path.exists(MODELS_FOLDER):
        os.mkdir(MODELS_FOLDER)

    # if init dataset is not given, create first model by training on random vs random dataset
    if INIT_DATASET == '' or INIT_DATASET is None:
        # generate init data random vs random
        print('Generating init Data...')
        t0 = time.time()
        boards, moves, _ = auto_rematch(random_move, random_move, args_1=tuple({True}), args_2=tuple({True}),
                                        n_matches=N_MATCHES)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

        # prepare data
        print('Preparing Data...')
        t0 = time.time()
        X, y = np.array(boards), np.array(moves)
    else:
        # load init dataset
        print(f'Preparing Data {INIT_DATASET}...')
        t0 = time.time()
        data = scipy.io.loadmat(INIT_DATASET)
        X, y = data['data'], data['labels'][0, :]

    # extract final boards, insert flipped boards, clean, encode, split
    X, y = extract_one_to_win(X, y)
    X, y = insert_flipped_boards(X, y)
    X, y = clean_data(X, y)

    # save as first growing iteration
    scipy.io.savemat(DATA_FOLDER + '0_GROWING.mat', {'data': X, 'labels': y})

    # encoding
    X = OneHotEncoder(categories=[[-1, 0, 1]] * 42).fit_transform(X).toarray().astype(np.int8)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    td = time.time()
    print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

    # train on data
    print('Training MLP...')
    print(f'Training on {len(y_train)} entries...')
    t0 = time.time()
    mlp = get_mlp().fit(X_train, y_train)
    td = time.time()
    print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')
    print('MLP score (Train):', mlp.score(X_train, y_train))
    print('MLP score (Test): ', mlp.score(X_test, y_test))

    # save model
    joblib.dump(mlp, MODELS_FOLDER + '0_' + MODEL_PATH)

    ### START ITERATIVE PROCESS
    for i in range(N_ITER):
        print('-' * 50)
        print('\nITERATION', i+1)
        # load new mlp
        mlp = joblib.load(MODELS_FOLDER + str(i) + '_' + MODEL_PATH)

        # generate data (random vs mlp)
        print('Generating Data...')
        t0 = time.time()
        boards, moves, a_wins = auto_rematch(mlp_move, random_move, args_1=tuple({mlp}), args_2=tuple({True}),
                                             n_matches=N_MATCHES, sa_ratio=SA_RATIO)
        print(f'{a_wins[PLAYER1]} out of {a_wins[PLAYER1] + a_wins[PLAYER2]} won')
        # save_data(boards, moves, filename=DATA_FOLDER + str(i+1) + '_' + DATASET)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

        # prepare data
        print('Preparing Data...')
        t0 = time.time()
        X_new, y_new = np.array(boards), np.array(moves)

        # extract final boards and moves
        X_new, y_new = extract_one_to_win(X_new, y_new)

        # extend growing data set
        data_grow = scipy.io.loadmat(DATA_FOLDER + str(i) + '_GROWING.mat')
        X, y = data_grow['data'], data_grow['labels'][0, :]
        X, y = np.concatenate((X, X_new), axis=0), np.concatenate((y, y_new), axis=0)

        # insert flipped boards and clean
        X, y = insert_flipped_boards(X, y)
        X, y = clean_data(X, y)

        # save extended dataset
        scipy.io.savemat(DATA_FOLDER + str(i+1) + '_GROWING.mat', {'data': X, 'labels': y})
        print(f'{len(y)} entries in total')

        # encoding
        X = OneHotEncoder(categories=[[-1, 0, 1]] * 42).fit_transform(X).toarray().astype(np.int8)

        # split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

        # train on data
        print('Training MLP...')
        print(f'Training on {len(y_train)} entries...')
        t0 = time.time()
        mlp.fit(X_train, y_train)
        td = time.time()
        print(f'\t\t\t\t\t{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')
        print('MLP score (Train):', mlp.score(X_train, y_train))
        print('MLP score (Test): ', mlp.score(X_test, y_test))

        # save model
        joblib.dump(mlp, MODELS_FOLDER + str(i+1) + '_' + MODEL_PATH)

    tdd = time.time()
    print(f'\nFinished after {((tdd - t00) // 3600):.0f} h {(((tdd - t00) // 60) % 60):.0f} min {((tdd - t00) % 60):.1f} sec')
