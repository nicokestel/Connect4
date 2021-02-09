from typing import Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier

from agents.common import BoardPiece, SavedState, PlayerAction, get_non_full_columns, PLAYER2, PLAYER1, \
    initialize_game_state

from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder(categories=[[-1, 0, 1]] * 42).fit(initialize_game_state().flatten().reshape(1, -1))

def generate_move_mlp(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], mlp: MLPClassifier
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Return move determined by MLP.

    Parameters
    ----------
    board : np.ndarray
        Current board
    player : BoardPiece
        Current player
    saved_state : Optional[SavedState]
        ???
    mlp : MLPClassifier
        Classifier that predicts next move

    Returns
    -------
    action : PlayerAction
        The generated move as index of column that is to be played
    saved_state : Optional[SavedState]
        ???
    """

    org_board = board.copy()

    # replace own pieces with 1, enemy pieces with -1
    board[board == (PLAYER2 if player == PLAYER1 else PLAYER1)] = -1
    board[board == player] = 1

    # flatten
    board = board.flatten().reshape(1, -1)

    # one hot encoding
    board = ohe.transform(board).toarray()

    # predict probabilities
    probs = mlp.predict_proba(board)[0, :]

    # set probabilities of full columns to zero
    probs *= np.isin(np.arange(0, 7), get_non_full_columns(org_board))

    # return column with highest probability
    action = PlayerAction(np.argmax(probs))

    return action, saved_state
