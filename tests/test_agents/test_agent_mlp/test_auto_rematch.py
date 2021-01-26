def test_auto_rematch():
    from agents.agent_mlp.mlp_training.auto_rematch import auto_rematch
    from agents.agent_random import generate_move as random_move
    import numpy as np

    n_matches = 100

    boards, moves = auto_rematch(random_move, random_move, n_matches=n_matches)

    # two games per match and at least four moves to win
    assert len(moves) == len(boards) >= 4 * 2 * n_matches

    # every single value has to be -1, 0 or 1
    assert np.all((np.array(boards) == -1) + (np.array(boards) == 0) + (np.array(boards) == 1))

    assert type(boards[0]) == np.ndarray
    assert type(boards[0][0]) == np.int8
    assert type(moves[0]) == np.int8
