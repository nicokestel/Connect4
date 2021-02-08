from typing import Tuple, Optional

import numpy as np

from ..common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, check_end_state, GameState, \
    get_non_full_columns, apply_player_action

Depth = np.int8  # data type of depth integer
MIN_VALUE = np.iinfo(np.int32).min
MAX_VALUE = np.iinfo(np.int32).max
DRAW_VALUE = np.int32(0)


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: Depth = Depth(4),
        use_ab_pruning: bool = True
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
    depth: Depth
        Indicates up to which depth moves should be predicted. (default = 4)
    use_ab_pruning: bool
        Flag that indicates if alphabeta-pruning should be used

    Returns
    -------
    action : PlayerAction
        The generated move as index of column that is to be played
    saved_state : Optional[SavedState]
        ???

    """

    if use_ab_pruning:
        return tuple(np.array(minimax_ab(board=board, player=player, depth=depth, saved_state=saved_state))[[0, 2]])

    return tuple(np.array(minimax(board=board, player=player, depth=depth, saved_state=saved_state))[[0, 2]])


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

    if depth == 0 or not check_end_state(board=board, player=player) == GameState.STILL_PLAYING:
        return PlayerAction(0), score(board=board), saved_state

    if player == PLAYER1:  # maximizing
        value = np.iinfo(np.int32).min
        moves = sort_moves(get_non_full_columns(board=board))
        best_move = moves[0]
        for move in moves:
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
        moves = get_non_full_columns(board=board)
        best_move = moves[0]
        for move in moves:
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
        board: np.ndarray, player: BoardPiece, depth: np.int8, saved_state: Optional[SavedState],
        a: np.int32 = MIN_VALUE, b: np.int32 = MAX_VALUE
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

    if depth == 0 or not check_end_state(board=board, player=player) == GameState.STILL_PLAYING:
        return PlayerAction(0), score(board=board), saved_state

    if player == PLAYER1:  # maximizing
        value = a
        moves = get_non_full_columns(board=board)
        best_move = moves[0]
        for move in moves:
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
        moves = sort_moves(get_non_full_columns(board=board))
        best_move = moves[0]
        for move in moves:
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


score_matrix = np.array([[3, 4, 5, 7, 5, 4, 3],
                         [4, 6, 8, 10, 8, 6, 4],
                         [5, 8, 11, 13, 11, 8, 5],
                         [5, 8, 11, 13, 11, 8, 5],
                         [4, 6, 8, 10, 8, 6, 4],
                         [3, 4, 5, 7, 5, 4, 3]], dtype=np.int32)


def score(board: np.ndarray) -> np.int32:
    """
    Scores a board using simple heuristic scoring.

    Parameters
    ----------
    board: np.ndarray
        Current game state

    Returns
    -------
    score: np.int32
        Heuristic score of board
        score = 0 := draw
        score > 0 := advantage for PLAYER1
        score < 0 := advantage for PLAYER2
        score = INF := PLAYER1 won
        score = -INF := PLAYER2 won

    """

    # check edge cases
    if check_end_state(board, PLAYER1) == GameState.IS_DRAW:
        return DRAW_VALUE

    if check_end_state(board, PLAYER1) == GameState.IS_WIN:
        return MAX_VALUE

    if check_end_state(board, PLAYER2) == GameState.IS_WIN:
        return MIN_VALUE

    # ongoing game
    player1_score = np.multiply((board == PLAYER1).astype(np.int32), score_matrix).sum()
    player2_score = np.multiply((board == PLAYER2).astype(np.int32), score_matrix).sum()

    return player1_score - player2_score


def sort_moves(moves: Tuple[PlayerAction]) -> Tuple[PlayerAction]:
    """
    Sorts moves according to distance to middle column (ascending).

    Parameters
    ----------
    moves : Tuple[PlayerAction]
        Moves in ascending order
    Returns
    -------
    moves : Tuple[PlayerAction]
        Sorted moves
    """
    m = np.array(moves)
    diff = abs(m - 3)
    idx = np.argsort(diff)

    return tuple(m[idx].astype(PlayerAction))


def feature_score(board: np.ndarray) -> np.int32:
    """
    Scores a board by looking for features.

    Parameters
    ----------
    board: np.ndarray
        Current game state

    Returns
    -------
    score: np.int32
        Heuristic feature score of board
        score = 0 := draw
        score > 0 := advantage for PLAYER1
        score < 0 := advantage for PLAYER2
        score = INF := PLAYER1 won
        score = -INF := PLAYER2 won

    """

    # check edge cases
    if check_end_state(board, PLAYER1) == GameState.IS_DRAW:
        return DRAW_VALUE

    if check_end_state(board, PLAYER1) == GameState.IS_WIN:
        return MAX_VALUE

    if check_end_state(board, PLAYER2) == GameState.IS_WIN:
        return MIN_VALUE

    masked_board = np.where(board == 2, -1, board)
    feat2_row_player1 = np.array([0, 1, 1, 1, 0])
    feat2_row_player2 = np.array([0, -1, -1, -1, 0])
    # split board into rows and columns
    rows = np.vsplit(masked_board, 6)
    cols = np.hsplit(masked_board, 7)

    # check for three in a row with a gap on both sides - certain win
    for i in range(6):
        # get indices of columns with a player 1 piece in them
        idx_player1 = np.where(rows[i][0] == 1)[0]
        idx_player2 = np.where(rows[i][0] == -1)[0]
        if len(idx_player1) == 0:
            continue
        else:
            for j, k in zip(idx_player1, idx_player2):
                # splice row at given index, and row below
                check_player1 = rows[i][0][j-1:j+4]
                check_below_player1 = rows[i+1][0][j-1:j+4]
                check_player2 = rows[i][0][k-1:k+4]
                check_below_player2 = rows[i + 1][0][k - 1:k + 4]
                if np.array_equal(check_player1, feat2_row_player1):
                    # check edge case: because of identical gaps in the row below, piece will not form connect four
                    # if played
                    if check_below_player1[0] == 0 or check_below_player1[-1] == 0:
                        return np.int32(0)
                    else:
                        return MAX_VALUE
                if np.array_equal(check_player2, feat2_row_player2):
                    # check edge case
                    if np.array_equal(check_below_player2, feat2_row_player1) or np.array_equal(check_player1,
                                                                                                feat2_row_player2):
                        return np.int32(0)
                    else:
                        return MAX_VALUE

    return np.int32(-1)
