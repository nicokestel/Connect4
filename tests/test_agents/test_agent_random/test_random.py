def test_generate_move_random():
    from agents.common import PlayerAction, PLAYER2, string_to_board
    from agents.agent_random import generate_move

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
