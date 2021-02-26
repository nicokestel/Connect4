import numpy as np

from ..common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, check_end_state, GameState, \
    get_non_full_columns, apply_player_action
from typing import Tuple, Optional

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
        return PlayerAction(0), score(board=board, f_score=True), saved_state

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


def score(board: np.ndarray, f_score: bool = False) -> np.int32:
    """
    Scores a board using simple heuristic scoring.

    Parameters
    ----------
    board: np.ndarray
        Current game state
    f_score: bool
        Flag for feature score heuristic. If False, use simple heuristic
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
    if f_score:
        return feature_score(board)

    game_state_player1 = check_end_state(board, PLAYER1)
    game_state_player2 = check_end_state(board, PLAYER2)

    if game_state_player1 == GameState.STILL_PLAYING \
       and game_state_player2 == GameState.STILL_PLAYING:
        # ongoing game
        player1_score = np.multiply((board == PLAYER1).astype(np.int32), score_matrix).sum()
        player2_score = np.multiply((board == PLAYER2).astype(np.int32), score_matrix).sum()

        return player1_score - player2_score

    # check edge cases
    if game_state_player1 == GameState.IS_DRAW:
        return DRAW_VALUE

    if game_state_player1 == GameState.IS_WIN:
        return MAX_VALUE

    if game_state_player2 == GameState.IS_WIN:
        return MIN_VALUE

    return DRAW_VALUE


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
    Scores a board by looking for features indicating strong winning scenarios e.g. vertical and
    horizontal permutations of three player pieces with one or two gaps to play in.

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

    # feature 1 - check win or draw cases
    if check_end_state(board, PLAYER1) == GameState.IS_DRAW:
        return DRAW_VALUE

    if check_end_state(board, PLAYER1) == GameState.IS_WIN:
        return MAX_VALUE

    if check_end_state(board, PLAYER2) == GameState.IS_WIN:
        return MIN_VALUE

    # mask and split board into rows, columns
    masked_board = np.where(board == 2, -1, board)
    rows = np.vsplit(masked_board, 6)
    cols = np.hsplit(masked_board, 7)

    # feature 2.1 - check for three in a row with a gap on both sides - certain win
    feat21_row_player1 = np.array([0, 1, 1, 1, 0])
    feat21_row_player2 = np.array([0, -1, -1, -1, 0])

    # loop over rows of board
    for i in range(6):
        # get indices of where player pieces are in each row
        idx_row_player1 = np.where(rows[i][0] == 1)[0]

        if len(idx_row_player1) == 0:
            continue
        else:
            for j in idx_row_player1:
                check_player1 = rows[i][0][j - 1:j + 4]
                if i == 5:
                    check_below_player1 = [3]  # 3 as a dummy value for "below" last row of the board
                else:
                    check_below_player1 = rows[i + 1][0][j - 1:j + 4]

                # check if feature exists for player 1
                if np.array_equal(check_player1, feat21_row_player1):
                    # check edge case: because of identical gaps in the row below, piece will not form connected four
                    # if played
                    if check_below_player1[0] == 0 and check_below_player1[-1] == 0:
                        continue
                    else:
                        return MAX_VALUE

        idx_row_player2 = np.where(rows[i][0] == -1)[0]
        if len(idx_row_player2) == 0:
            continue
        else:
            for k in idx_row_player2:
                check_player2 = rows[i][0][k - 1:k + 4]
                if i == 5:
                    check_below_player2 = [3]
                else:
                    check_below_player2 = rows[i + 1][0][k - 1:k + 4]
                    # check if feature exists for player 2
                if np.array_equal(check_player2, feat21_row_player2):
                    # check edge case
                    if check_below_player2[0] == 0 and check_below_player2[-1] == 0:
                        continue
                    else:
                        return MIN_VALUE

    # initialize heuristic value
    heuristic_value = np.int32(0)

    # feature 2.2 - check for row of 3 pieces with a gap of one, or column of 3 with a gap above
    feat22_row1_player1 = np.array([0, 1, 1, 1])
    feat22_row2_player1 = np.array([1, 0, 1, 1])
    feat22_row3_player1 = np.array([1, 1, 0, 1])
    feat22_row4_player1 = np.array([1, 1, 1, 0])
    feat22_row1_player2 = np.array([0, -1, -1, -1])
    feat22_row2_player2 = np.array([-1, 0, -1, -1])
    feat22_row3_player2 = np.array([-1, -1, 0, -1])
    feat22_row4_player2 = np.array([-1, -1, -1, 0])

    # loop over rows in the board
    for i in range(6):
        # get rows with player pieces in them
        idx_row_player1 = np.where(rows[i][0] == 1)[0]
        idx_row_player2 = np.where(rows[i][0] == -1)[0]

        # check features for player 1
        if len(idx_row_player1) == 0:
            continue
        for j in idx_row_player1:
            check_feat221_player1 = rows[i][0][j - 1:j + 3]
            check_feat222_player1 = rows[i][0][j:j + 4]

            if i == 5:
                # dummy value for the row "below" last row of the board
                check_below_row1_player1 = [3, 3, 3, 3]
                check_below_row2_player1 = [3, 3, 3, 3]
            else:
                check_below_row1_player1 = rows[i + 1][0][j - 1:j + 3]
                check_below_row2_player1 = rows[i + 1][0][j:j + 4]

            if np.array_equal(feat22_row1_player1, check_feat221_player1):
                if check_below_row1_player1[0] == 0:
                    pass
                else:
                    heuristic_value += 900000

            if np.array_equal(feat22_row2_player1, check_feat222_player1):
                if check_below_row2_player1[1] == 0:
                    pass
                else:
                    heuristic_value += 900000

            if np.array_equal(feat22_row3_player1, check_feat222_player1):
                if check_below_row2_player1[1] == 0:
                    pass
                else:
                    heuristic_value += 900000

            if np.array_equal(feat22_row4_player1, check_feat222_player1):
                if check_below_row2_player1[1] == 0:
                    pass
                else:
                    heuristic_value += 900000

        # check features for player 2
        if len(idx_row_player2) == 0:
            continue
        for k in idx_row_player2:  # j - player 1 index, k - player 2 index
            # splice row at the index to an array of length matching the feature we want to check (for each player)
            check_feat221_player2 = rows[i][0][k - 1:k + 3]
            check_feat222_player2 = rows[i][0][k:k + 4]

            if i == 5:  # dummy value for the row below if we are checking the last row
                check_below_row1_player2 = [3, 3, 3, 3]
                check_below_row2_player2 = [3, 3, 3, 3]
            else:
                check_below_row1_player2 = rows[i + 1][0][k - 1:k + 3]
                check_below_row2_player2 = rows[i + 1][0][k:k + 4]

            # check if feature 2.1 exists for each player

            if np.array_equal(feat22_row1_player2, check_feat221_player2):
                if check_below_row1_player2[0] == 0:
                    pass
                else:
                    heuristic_value -= 900000

            if np.array_equal(feat22_row2_player2, check_feat222_player2):
                if check_below_row2_player2[1] == 0:
                    pass
                else:
                    heuristic_value -= 900000

            if np.array_equal(feat22_row3_player2, check_feat222_player2):
                if check_below_row2_player2[1] == 0:
                    pass
                else:
                    heuristic_value -= 900000

            if np.array_equal(feat22_row4_player2, check_feat222_player2):
                if check_below_row2_player2[1] == 0:
                    pass
                else:
                    heuristic_value -= 900000

    # feature 2 - three in a row, vertical
    feat22_col_player1 = feat22_row1_player1.reshape(4, 1)
    feat22_col_player2 = feat22_row1_player2.reshape(4, 1)

    # loop over columns of board
    for i in range(7):
        # get columns with pieces played
        idx_col_player1 = np.where(cols[i] == 1)[0]
        idx_col_player2 = np.where(cols[i] == -1)[0]

        # loop through columns, searching player 1 pieces
        for j in idx_col_player1:
            if j > 3:  # vertical column of length four only possible at depth 4 or higher in board
                check_col_player1 = cols[i][j-3:j+1]
                if np.array_equal(feat22_col_player1, check_col_player1):
                    heuristic_value += 900000
        # loop through columns, searching for player 2 pieces
        for k in idx_col_player2:
            if k > 3:
                check_col_player2 = cols[i][k - 3:k + 1]
                if np.array_equal(feat22_col_player2, check_col_player2):
                    heuristic_value -= 900000

    return heuristic_value
