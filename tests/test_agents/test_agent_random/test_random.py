from agents.common import *
from typing import Tuple
from agents.agent_random import generate_move, get_non_full_columns
import numpy as np


def test_get_non_full_columns():
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


def test_generate_move_random():
    board_end = "|==============|\n" \
                "|O O O X O O   |\n" \
                "|X X X O X X X |\n" \
                "|O O X X X O X |\n" \
                "|X X O O X X O |\n" \
                "|O O X X O O X |\n" \
                "|O O X X O X O |\n" \
                "|==============|\n" \
                "|0 1 2 3 4 5 6 |"
    gen_move = generate_move(board=string_to_board(board_end), player=PLAYER2, saved_state=None)[0]

    assert isinstance(gen_move, PlayerAction)
    assert gen_move == 6
