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

    """
    Minimax pseudocode:
    
    function minimax(board, depth, isMaxPlayer)
        if depth=0 or terminal node
            return score(board)
        
        if isMaxPlayer=TRUE
            value := -INF
            for each child
                value := max(value, minimax(child, depth-1, FALSE)
            return value
         else 
            value := INF
            for each child
                value := min(value, minimax(child, depth-1, TRUE)
            return value
    
    """

    """
        Minimax + AlphaBeta-Pruning pseudocode:

        function minimax(board, depth, a, b, isMaxPlayer)
            if depth=0 or terminal node
                return score(board)

            if isMaxPlayer=TRUE
                value := a
                
                for each child
                    value := max(value, minimax(child, depth-1, value, b, FALSE)
                    
                    if value >= b   BETA-CUTOFF
                        break
                        
                return value
                
             else 
                value := b
                
                for each child
                    value := min(value, minimax(child, depth-1, a, value, TRUE)
                    
                    if value <= a   ALPHA-CUTOFF
                        break
                        
                return value

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

    # should respect GameState.IS_DRAW with return value 0
    # should respect GameState.IS_WIN with very high/low values

    # for 'normal' situation consider evaluating the board from
    # both perspectives and returning the difference.

    # A larger difference is equivalent to a bigger advantage
    # for the current player

    pass
