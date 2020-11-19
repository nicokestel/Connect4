import numpy as np
from ..common import BoardPiece, SavedState, PlayerAction
from typing import Tuple, Optional


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

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
    pass


def minimax(
        board: np.ndarray, player: BoardPiece, depth: np.int8, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

    Parameters
    ----------
    board: np.ndarray
        Current game state
    player: BoardPiece
        The player that needs to play a move
    depth: np.int8
        Depth counter to indicate how many turns are left to predict (stopping at 0)
    saved_state: Optional[SavedState]
        ???


    Returns
    -------
    action : PlayerAction
        The generated move as index of column that is to be played
    saved_state : Optional[SavedState]
        ???

    """
    pass


def score(board: np.ndarray, player: BoardPiece) -> np.int32:
    """
    Scores a board from a player's perspective using a heuristic.

    Parameters
    ----------
    board: np.ndarray
        Current game state
    player: BoardPiece
        The player's perspective from which to score the board

    Returns
    -------
    score: np.int32
        Heuristic score of board from player's perspective

    """
    pass
