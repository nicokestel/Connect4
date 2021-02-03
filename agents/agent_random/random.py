import numpy as np
from ..common import BoardPiece, SavedState, PlayerAction, get_non_full_columns
from typing import Tuple, Optional


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], uniform_distribution: bool = False
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
    uniform_distribution : bool
        True: Moves are equally likely
        False (default): Moves tend to center column

    Returns
    -------
    action : PlayerAction
        The generated move as index of column that is to be played
    saved_state : Optional[SavedState]
        ???

    """

    # Choose a valid, non-full column randomly and return it as 'action'
    if uniform_distribution:
        return np.random.choice(get_non_full_columns(board)), saved_state

    p = np.array([1/16, 1/8, 3/16, 1/4, 3/16, 1/8, 1/16])
    available = np.isin(np.arange(0, 7), get_non_full_columns(board))

    p *= available
    p /= sum(p)

    action = np.random.choice(np.arange(0, 7), p=p)

    return action, saved_state
