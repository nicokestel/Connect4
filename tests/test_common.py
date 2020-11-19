from agents.common import ROWS, COLUMNS, BoardPiece, NO_PLAYER, PLAYER2, PLAYER1, initialize_game_state, PlayerAction
from typing import Tuple
import numpy as np


def test_initialize_game_state():
    board = initialize_game_state()

    assert isinstance(board, np.ndarray)
    assert board.shape == (ROWS, COLUMNS)
    assert board.dtype == BoardPiece
    assert board.all() == NO_PLAYER


def test_pretty_print_board():
    from agents.common import pretty_print_board

    board = initialize_game_state()
    board[0, 1] = PLAYER2
    board[1, 1] = PLAYER2

    board[0, 2] = PLAYER2
    board[1, 2] = PLAYER1
    board[2, 2] = PLAYER2
    board[3, 2] = PLAYER1

    board[0, 3] = PLAYER1
    board[1, 3] = PLAYER2
    board[2, 3] = PLAYER1
    board[3, 3] = PLAYER1

    board[0, 4] = PLAYER1
    board[1, 4] = PLAYER2
    board[2, 4] = PLAYER1

    cmp_board = "|==============|\n" \
                "|              |\n" \
                "|              |\n" \
                "|    X X       |\n" \
                "|    O X X     |\n" \
                "|  O X O O     |\n" \
                "|  O O X X     |\n" \
                "|==============|\n" \
                "|0 1 2 3 4 5 6 |"

    assert pretty_print_board(board) == cmp_board


def test_string_to_board():
    from agents.common import string_to_board

    input_board = "|==============|\n" \
                  "|              |\n" \
                  "|              |\n" \
                  "|    X X       |\n" \
                  "|    O X X     |\n" \
                  "|  O X O O     |\n" \
                  "|  O O X X     |\n" \
                  "|==============|\n" \
                  "|0 1 2 3 4 5 6 |"

    cmp_board = initialize_game_state()
    cmp_board[0, 1] = PLAYER2
    cmp_board[1, 1] = PLAYER2

    cmp_board[0, 2] = PLAYER2
    cmp_board[1, 2] = PLAYER1
    cmp_board[2, 2] = PLAYER2
    cmp_board[3, 2] = PLAYER1

    cmp_board[0, 3] = PLAYER1
    cmp_board[1, 3] = PLAYER2
    cmp_board[2, 3] = PLAYER1
    cmp_board[3, 3] = PLAYER1

    cmp_board[0, 4] = PLAYER1
    cmp_board[1, 4] = PLAYER2
    cmp_board[2, 4] = PLAYER1

    assert string_to_board(input_board).shape == (ROWS, COLUMNS)
    assert np.allclose(string_to_board(input_board), cmp_board)


def test_apply_player_action():
    from agents.common import apply_player_action

    cmp_board = initialize_game_state()
    cmp_board[0, 0] = PLAYER2
    cmp_board[0, 1] = PLAYER1
    cmp_board[0, 2] = PLAYER2
    cmp_board[1, 1] = PLAYER1
    cmp_board[2, 1] = PLAYER2
    cmp_board[3, 1] = PLAYER2
    cmp_board[4, 1] = PLAYER2
    cmp_board[5, 1] = PLAYER2

    board = initialize_game_state()
    board = apply_player_action(board=board, action=PlayerAction(0), player=PLAYER2)
    board = apply_player_action(board=board, action=PlayerAction(1), player=PLAYER1)
    board = apply_player_action(board=board, action=PlayerAction(2), player=PLAYER2)
    board = apply_player_action(board=board, action=PlayerAction(1), player=PLAYER1)
    board = apply_player_action(board=board, action=PlayerAction(1), player=PLAYER2)
    board = apply_player_action(board=board, action=PlayerAction(1), player=PLAYER2)
    board = apply_player_action(board=board, action=PlayerAction(1), player=PLAYER2)
    board = apply_player_action(board=board, action=PlayerAction(1), player=PLAYER2)
    # test column capacity
    board = apply_player_action(board=board, action=PlayerAction(1), player=PLAYER2)

    assert np.allclose(board, cmp_board)

    # test copy
    copy_board = apply_player_action(board=board, action=PlayerAction(6), player=PLAYER1,
                                     copy=True)

    assert not np.allclose(board, copy_board)


def test_get_non_full_columns():
    from agents.common import string_to_board, get_non_full_columns

    board = initialize_game_state()
    assert isinstance(get_non_full_columns(board), Tuple)
    assert isinstance(get_non_full_columns(board)[0], PlayerAction)
    assert get_non_full_columns(board) == (0, 1, 2, 3, 4, 5, 6)

    board2 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X       |\n" \
             "|    O X X     |\n" \
             "|  O X O O     |\n" \
             "|  O O X X     |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"
    assert get_non_full_columns(string_to_board(board2)) == (0, 1, 2, 3, 4, 5, 6)

    board3 = "|==============|\n" \
             "|    X         |\n" \
             "|    O O       |\n" \
             "|    X X       |\n" \
             "|    O X X     |\n" \
             "|  O X O O     |\n" \
             "|  O O X X     |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"
    assert get_non_full_columns(string_to_board(board3)) == (0, 1, 3, 4, 5, 6)

    board_draw = "|==============|\n" \
                 "|O O O X O O O |\n" \
                 "|X X X O X X X |\n" \
                 "|O O X X X O X |\n" \
                 "|X X O O X X O |\n" \
                 "|O O X X O O X |\n" \
                 "|O O X X O X O |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |"
    assert get_non_full_columns(string_to_board(board_draw)) == ()


def test_connected_four():
    from agents.common import string_to_board, connected_four

    board1 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X       |\n" \
             "|    O X X     |\n" \
             "|  O X O O     |\n" \
             "|  O O X X     |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"

    board2 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X       |\n" \
             "|    X X X     |\n" \
             "|  O X O O     |\n" \
             "|  O X X X     |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"

    board3 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|  O X X       |\n" \
             "|  X O X X     |\n" \
             "|  O X O O     |\n" \
             "|  O O X O     |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"

    board4 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X     X |\n" \
             "|    O X X X O |\n" \
             "|  O X O X O O |\n" \
             "|  O O X X O O |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"

    board5 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X     X |\n" \
             "|    O X X X O |\n" \
             "|  O X O X O O |\n" \
             "|  O O X X X O |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"

    board_draw = "|==============|\n" \
                 "|O O O X O O O |\n" \
                 "|X X X O X X X |\n" \
                 "|O O X X X O X |\n" \
                 "|X X O O X X O |\n" \
                 "|O O X X O O X |\n" \
                 "|O O X X O X O |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |"

    board_full = "|==============|\n" \
                 "|O O X O O O O |\n" \
                 "|X X X O X X X |\n" \
                 "|O O X X X O X |\n" \
                 "|X X O O X X O |\n" \
                 "|O O X X O O X |\n" \
                 "|O O X X O X O |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |"

    assert not connected_four(string_to_board(board1), PLAYER1)
    assert connected_four(string_to_board(board2), PLAYER1)
    assert connected_four(string_to_board(board3), PLAYER2)
    assert connected_four(string_to_board(board4), PLAYER1)

    # test last action
    assert connected_four(string_to_board(board5), PLAYER1, last_action=PlayerAction(5))
    assert connected_four(string_to_board(board5), PLAYER1, last_action=PlayerAction(2))
    assert not connected_four(string_to_board(board5), PLAYER1, last_action=PlayerAction(4))

    assert not connected_four(string_to_board(board_draw), PLAYER1)
    assert not connected_four(string_to_board(board_draw), PLAYER2)

    assert not connected_four(string_to_board(board_full), PLAYER1)
    assert connected_four(string_to_board(board_full), PLAYER2)


def test_check_end_state():
    from agents.common import string_to_board, GameState, check_end_state

    board1 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X       |\n" \
             "|    O X X     |\n" \
             "|  O X O O     |\n" \
             "|  O O X X     |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"

    board2 = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X       |\n" \
             "|    X X X     |\n" \
             "|  O X O O     |\n" \
             "|  O X X X     |\n" \
             "|==============|\n" \
             "|0 1 2 3 4 5 6 |"

    board3 = board2.replace('X', 'T').replace('O', 'X').replace('T', 'O')  # flip Xs and Os

    board_draw = "|==============|\n" \
                 "|O O O X O O O |\n" \
                 "|X X X O X X X |\n" \
                 "|O O X X X O X |\n" \
                 "|X X O O X X O |\n" \
                 "|O O X X O O X |\n" \
                 "|O O X X O X O |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |"

    assert check_end_state(string_to_board(board_draw), PLAYER1) == GameState.IS_DRAW
    assert check_end_state(string_to_board(board_draw), PLAYER2,
                           PlayerAction(4)) == GameState.IS_DRAW
    assert check_end_state(string_to_board(board1), PLAYER1) == GameState.STILL_PLAYING
    assert check_end_state(string_to_board(board1), PLAYER2) == GameState.STILL_PLAYING
    assert check_end_state(string_to_board(board2), PLAYER1) == GameState.IS_WIN
    assert check_end_state(string_to_board(board3), PLAYER2) == GameState.IS_WIN

    assert check_end_state(string_to_board(board2), PLAYER1,
                           PlayerAction(2)) == GameState.IS_WIN
    assert check_end_state(string_to_board(board2), PLAYER1,
                           PlayerAction(3)) == GameState.STILL_PLAYING

    assert check_end_state(string_to_board(board3), PLAYER2) == GameState.IS_WIN
