import numpy as np
from ..common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, check_end_state, GameState, \
    get_non_full_columns, apply_player_action
from typing import Tuple, Optional

Depth = np.int8  # data type of depth integer


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
) -> Tuple[PlayerAction, np.int32, Optional[SavedState]]:
    """
    Executes Minimax Algorithm from player's perspective by predicting depth next turns starting with board.
    Returns best column to play.

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
    value : np.int32
        Value of generated move
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
    if depth == 0 or not check_end_state(board=board, player=player) == GameState.STILL_PLAYING:
        return PlayerAction(0), score(board=board, player=player), saved_state

    if player == PLAYER1:  # maximizing
        value = np.iinfo(np.int32).min
        best_move = None
        for move in get_non_full_columns(board=board):
            (_, new_value, _) = minimax(board=apply_player_action(board=board,
                                                                  action=move,
                                                                  player=player,
                                                                  copy=True),  # 'simulate' move
                                        player=PLAYER2,  # minimizing player
                                        depth=depth - 1,
                                        saved_state=saved_state)

            if new_value > value:
                best_move = move
                value = new_value

        return best_move, value, saved_state

    else:  # PLAYER2 minimizing
        value = np.iinfo(np.int32).max
        best_move = None
        for move in get_non_full_columns(board=board):
            (_, new_value, _) = minimax(board=apply_player_action(board=board,
                                                                  action=move,
                                                                  player=player,
                                                                  copy=True),  # 'simulate' move
                                        player=PLAYER1,  # maximizing player
                                        depth=depth - 1,
                                        saved_state=saved_state)

            if new_value < value:
                best_move = move
                value = new_value

        return best_move, value, saved_state


def minimax_ab(
        board: np.ndarray, player: BoardPiece, depth: np.int8, a: np.int32, b: np.int32,
        saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, np.int32, Optional[SavedState]]:
    """
    Executes Minimax Algorithm with alphabeta-pruning from player's perspective by predicting depth next turns
    starting with board.
    Returns best column to play.

    Parameters
    ----------
    board: np.ndarray
        Current game state
    player: BoardPiece
        The player that needs to play a move
    depth: np.int8
        Depth counter to indicate how many turns are left to predict (stopping at 0)
    a: np.int32
        Alpha value used for alpha-cutoffs
    b: np.int32
        Beta value used for beta-cutoffs
    saved_state: Optional[SavedState]
        ???


    Returns
    -------
    action : PlayerAction
        The generated move as index of column that is to be played
    value : np.int32
        Value of generated move
    saved_state : Optional[SavedState]
        ???

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

    if depth == 0 or not check_end_state(board=board, player=player) == GameState.STILL_PLAYING:
        return PlayerAction(0), score(board=board, player=player), saved_state

    if player == PLAYER1:  # maximizing
        value = a
        best_move = None
        for move in get_non_full_columns(board=board):
            (_, new_value, _) = minimax_ab(board=apply_player_action(board=board,
                                                                     action=move,
                                                                     player=player,
                                                                     copy=True),  # 'simulate' move
                                           player=PLAYER2,  # minimizing player
                                           depth=depth - 1,
                                           a=value,
                                           b=b,
                                           saved_state=saved_state)

            if new_value > value:
                best_move = move
                value = new_value

            if value >= b:  # BETA-cutoff
                break

        return best_move, value, saved_state

    else:  # PLAYER2 minimizing
        value = b
        best_move = None
        for move in get_non_full_columns(board=board):
            (_, new_value, _) = minimax_ab(board=apply_player_action(board=board,
                                                                     action=move,
                                                                     player=player,
                                                                     copy=True),  # 'simulate' move
                                           player=PLAYER1,  # maximizing player
                                           depth=depth - 1,
                                           a=a,
                                           b=value,
                                           saved_state=saved_state)

            if new_value < value:
                best_move = move
                value = new_value

            if value <= a:  # ALPHA-cutoff
                break

        return best_move, value, saved_state


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
    # e.g.  np.iinfo(np.int32).max
    #       np.iinfo(np.int32).min

    # for 'normal' situation consider evaluating the board from
    # both perspectives and returning the difference.

    # A larger difference is equivalent to a bigger advantage
    # for the current player
    if check_end_state(board, player) == GameState.IS_DRAW:
        return np.int32(0)

    score_matrix = np.array([[3, 4,  5,  7,  5, 4, 3],
                             [4, 6,  8, 10,  8, 6, 4],
                             [5, 8, 11, 13, 11, 8, 5],
                             [5, 8, 11, 13, 11, 8, 5],
                             [4, 6,  8, 10,  8, 6, 4],
                             [3, 4,  5,  7,  5, 4, 3]], dtype=np.int32)

    masked_board = (board == PLAYER1).astype(np.int32)
    res = np.multiply(masked_board, score_matrix).sum()
    return res
