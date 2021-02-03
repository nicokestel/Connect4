def test_minimax():
    from agents.agent_minimax.minimax import minimax, minimax_ab, Depth, MIN_VALUE, MAX_VALUE
    from agents.common import initialize_game_state, string_to_board, PLAYER1, PLAYER2, PlayerAction

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

    assert minimax(board=board_init, player=PLAYER1, depth=Depth(4), saved_state=None)[0] == PlayerAction(3)
    assert minimax_ab(board=board_init, player=PLAYER1, depth=Depth(4), a=MIN_VALUE, b=MAX_VALUE, saved_state=None)[
               0] == PlayerAction(3)

    # PLAYER1 plays 3 to win, PLAYER2 needs to prevent that
    assert minimax(board=string_to_board(board_xwin), player=PLAYER1, depth=Depth(2), saved_state=None)[
               0] == PlayerAction(3)
    assert minimax(board=string_to_board(board_xwin), player=PLAYER2, depth=Depth(2), saved_state=None)[
               0] == PlayerAction(3)
    assert minimax_ab(board=string_to_board(board_xwin), player=PLAYER1, depth=Depth(2), a=MIN_VALUE, b=MAX_VALUE,
                      saved_state=None)[
               0] == PlayerAction(3)
    assert minimax_ab(board=string_to_board(board_xwin), player=PLAYER2, depth=Depth(2), a=MIN_VALUE, b=MAX_VALUE,
                      saved_state=None)[
               0] == PlayerAction(3)

    # PLAYER1 wins in 2 turns (actually)
    assert minimax(board=string_to_board(board), player=PLAYER1, depth=Depth(4), saved_state=None)[0] == PlayerAction(2)
    assert minimax_ab(board=string_to_board(board), player=PLAYER1, depth=Depth(4), a=MIN_VALUE, b=MAX_VALUE,
                      saved_state=None)[
               0] == PlayerAction(2)


def test_score():
    from agents.agent_minimax.minimax import score, MAX_VALUE, MIN_VALUE
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

    board_flip = board.replace('X', 'T').replace('O', 'X').replace('T', 'O')  # flip Xs and Os

    board_draw = "|==============|\n" \
                 "|O O O X O O O |\n" \
                 "|X X X O X X X |\n" \
                 "|O O X X X O X |\n" \
                 "|X X O O X X O |\n" \
                 "|O O X X O O X |\n" \
                 "|O O X X O X O |\n" \
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

    # not sure if PLAYER1 or PLAYER2 is winning, but should not indicate a draw
    assert score(board=string_to_board(board)) == 24
    assert score(board=string_to_board(board_flip)) == -24

    # test for draw
    assert score(board=string_to_board(board_draw)) == 0
    assert score(board=string_to_board(board_draw)) == 0

    # test a win situation for player1
    assert score(board=string_to_board(board2)) == MAX_VALUE

    # test a win situation for player2
    board2_flip = board2.replace('X', 'T').replace('O', 'X').replace('T', 'O')  # flip Xs and Os
    assert score(board=string_to_board(board2_flip)) == MIN_VALUE


def test_sort_moves():
    from agents.agent_minimax.minimax import sort_moves
    from agents.common import PlayerAction

    moves1 = (0, 1, 2, 3, 4, 5, 6)
    assert sort_moves(moves1) == (3, 2, 4, 1, 5, 0, 6)

    moves2 = (2, 3, 4, 6)
    assert sort_moves(moves2) == (3, 2, 4, 6)

    assert type(sort_moves(moves2)) == tuple
    assert type(sort_moves(moves2)[0]) == PlayerAction

    moves3 = (0, 1, 2, 4, 5, 6)
    assert sort_moves(moves3) == (2, 4, 1, 5, 0, 6)
