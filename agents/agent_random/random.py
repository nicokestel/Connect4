import numpy as np
from ..common import BoardPiece, SavedState, PlayerAction, get_non_full_columns
from typing import Tuple, Optional


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates a random possible move for player on board.

    Parameters
    ----------
    board : np.ndarray
        Current game state
    player : BoardPiece
        The player that needs to play a move
    saved_state : Optional[SavedState]
        ???

    Returns
    -------
    action : PlayerAction
        The generated move as index of column that is to be played
    saved_state : Optional[SavedState]
        ???

    """

    # Choose a valid, non-full column randomly and return it as 'action'
    action = np.random.choice(get_non_full_columns(board))

    return action, saved_state
