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