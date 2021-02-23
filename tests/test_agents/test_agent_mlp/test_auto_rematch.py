from agents.common import PLAYER1, PLAYER2


def test_auto_rematch():
    from agents.agent_mlp.mlp_training.auto_rematch import auto_rematch
    from agents.agent_random import generate_move as random_move
    import numpy as np

    n_matches = 100

    boards, moves, a_wins = auto_rematch(random_move, random_move, n_matches=n_matches)

    # two games per match and at least four moves to win
    assert len(moves) == len(boards) >= 4 * 2 * n_matches

    # every single value has to be -1, 0 or 1
    assert np.all((np.array(boards) == -1) + (np.array(boards) == 0) + (np.array(boards) == 1))

    assert type(boards[0]) == np.ndarray
    assert type(boards[0][0]) == np.int8
    assert type(moves[0]) == np.int8

    assert a_wins[PLAYER1] + a_wins[PLAYER2] == 2 * n_matches

    # test second agent win ratio option
    sa_ratio = 0.3
    boards, moves, a_wins = auto_rematch(random_move, random_move, n_matches=n_matches, sa_ratio=sa_ratio)

    assert a_wins[PLAYER1] + a_wins[PLAYER2] == 2 * n_matches
    assert a_wins[PLAYER1] == 2 * n_matches * (1-sa_ratio)
    assert a_wins[PLAYER2] == 2 * n_matches * sa_ratio

    # further type checks
    assert type(boards) == list and type(moves) == list
    assert len(boards) != 0 and len(moves) != 0
    assert len(boards) == len(moves)
    for board in boards:
        for elem in board:
            assert elem in [-1, 1, 0]
