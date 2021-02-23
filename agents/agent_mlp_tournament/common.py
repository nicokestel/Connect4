from enum import Enum
from typing import Optional
from typing import Callable, Tuple
import numpy as np

COLUMNS = np.int8(7)  # number of columns on board
ROWS = np.int8(6)  # number of rows on board

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """

    board = np.tile(NO_PLAYER, (ROWS, COLUMNS)).astype(BoardPiece)
    return board


def get_non_full_columns(board: np.ndarray) -> Tuple[PlayerAction]:
    """
    Receives a board and return the indices of playable columns.

    Parameters
    ----------
    board : np.ndarray
        Current board

    Returns
    -------
    actions : Tuple[PlayerAction]
        Playable moves on board

    """

    actions = np.where(board[-1, :] == NO_PLAYER)[0].astype(PlayerAction)
    return tuple(actions)
