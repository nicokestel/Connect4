def test_clean_data():
    import scipy.io
    import numpy as np
    from agents.agent_mlp.mlp_training.train_mlp import clean_data

    data = scipy.io.loadmat('0_MLP_RA.mat')
    X, y = data['data'], data['labels'][0, :]

    X_unique, idx_inv = np.unique(X, axis=0, return_inverse=True)

    X_clean, y_clean = clean_data(X, y)

    assert X_clean.shape[0] == y_clean.shape[0] == X_unique.shape[0]

    # test move assignment for 5 random unique entries
    for _ in range(5):
        # test most played column
        random_unique_entry = np.random.choice(np.arange(0, X_clean.shape[0]))
        y_to_random_entry, counts = np.unique(y[idx_inv == random_unique_entry], return_counts=True)

        # most played column for random entry
        most_played_to_random_entry = y_to_random_entry[np.argmax(counts)]

        assert y_clean[random_unique_entry] == most_played_to_random_entry


def test_insert_flipped_boards():
    from agents.agent_mlp.mlp_training.train_mlp import insert_flipped_boards
    import scipy.io
    import numpy as np

    data = scipy.io.loadmat('0_MLP_RA.mat')
    X, y = data['data'], data['labels'][0, :]

    n_entries_org = len(y)

    Xnew, ynew = insert_flipped_boards(X, y)

    assert type(Xnew) == type(Xnew[0]) == np.ndarray
    assert type(Xnew[0, 0]) == np.int8

    assert type(ynew) == np.ndarray
    assert type(ynew[0]) == np.int8

    assert len(ynew) == Xnew.shape[0] >= n_entries_org


def test_extract_one_to_win():
    from agents.agent_mlp.mlp_training.train_mlp import extract_one_to_win
    from agents.common import PLAYER2, PLAYER1, apply_player_action, check_end_state, GameState
    import scipy.io
    import numpy as np

    data = scipy.io.loadmat('0_MLP_RA.mat')
    X, y = data['data'], data['labels'][0, :]

    Xnew, ynew = extract_one_to_win(X, y)

    Xnew[Xnew == -1] = PLAYER2

    X_unique, c_unique = np.unique(Xnew, return_counts=True, axis=0)

    assert X_unique.shape == Xnew.shape
    assert np.all(c_unique == 1)

    for i in range(Xnew.shape[0]):
        board = Xnew[i].reshape(6, 7)
        board = apply_player_action(board, ynew[i], PLAYER1)
        assert check_end_state(board, PLAYER2) != GameState.IS_WIN
