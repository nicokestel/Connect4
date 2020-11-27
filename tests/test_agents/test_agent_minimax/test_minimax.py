def test_minimax():
    from agents.agent_minimax.minimax import minimax
    from agents.common import initialize_game_state, string_to_board, PLAYER1, PLAYER2

    board_init = initialize_game_state()

    board_xwin = "|==============|\n" \
                 "|O O O         |\n" \
                 "|X X X         |\n" \
                 "|O O X X X O X |\n" \
                 "|X X O O X X O |\n" \
                 "|O O X X O O X |\n" \
                 "|O O X X O X O |\n" \
                 "|==============|\n" \
                 "|0 1 2 3 4 5 6 |"

    board = "|==============|\n" \
            "|              |\n" \
            "|        O X   |\n" \
            "|      O X O   |\n" \
            "|  X   X X O   |\n" \
            "|  O   X O X   |\n" \
            "|O O   X O X   |\n" \
            "|==============|\n" \
            "|0 1 2 3 4 5 6 |"

    assert minimax(board=board_init, player=PLAYER1, depth=4)[0] == 3

    # PLAYER1 plays 3 to win, PLAYER2 needs to prevent that
    assert minimax(board=string_to_board(board_xwin), player=PLAYER1, depth=2)[0] == 3
    assert minimax(board=string_to_board(board_xwin), player=PLAYER2, depth=2)[0] == 3

    # PLAYER1 wins in 2 turns
    assert minimax(board=string_to_board(board), player=PLAYER1, depth=4)[0] == 2


def test_score():
    from agents.agent_minimax.minimax import score
    from agents.common import string_to_board, PLAYER1, PLAYER2
    board = "|==============|\n" \
             "|              |\n" \
             "|              |\n" \
             "|    X X       |\n" \
             "|    O X X     |\n" \
             "|  O X O O     |\n" \
             "|  O O X X     |\n" \
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

    # not sure if PLAYER1 or PLAYER2 is winning, but should not indicate a draw
    assert score(board=string_to_board(board), player=PLAYER1) != 0
    assert score(board=string_to_board(board), player=PLAYER2) != 0

    # test for draw
    assert score(board=string_to_board(board_draw), player=PLAYER1) == 0
    assert score(board=string_to_board(board_draw), player=PLAYER2) == 0