import numpy as np
from agents.agent_random import generate_move as random_move
from agents.agent_minimax import generate_move as minimax_move
from agents.common import pretty_print_board

def test_auto_rematch():
    from agents.agent_mlp.mlp_training.auto_rematch import auto_rematch
    boards, moves = auto_rematch(random_move, random_move)

    assert type(boards) == list and type(moves) == list
    assert len(boards) != 0 and len(moves) != 0
    assert len(boards) == len(moves)

    # print for debugging
    #for board in boards:
    #    print(pretty_print_board(board))
    #    print("--------------")