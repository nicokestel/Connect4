from typing import List
import numpy as np
import scipy.io


def save_data(data: List[np.ndarray], labels: List[np.int8], filename: str = 'C4-mlp-data.mat') -> np.int64:
    """

    Parameters
    ----------
    data : List[np.ndarray]
        Data to save
    labels : List[np.int8]
        Labels to save
    filename: str
        Filename

    Returns
    -------
    entries: np.int64
        Number of entries
    """
    scipy.io.savemat(filename, {'data': data, 'labels': labels})
    return len(labels)


if __name__ == '__main__':
    from agents.agent_mlp.mlp_training.auto_rematch import auto_rematch
    from agents.agent_minimax import generate_move as mm_move
    from agents.agent_random import generate_move as random_move
    from agents.agent_mlp import generate_move as mlp_move
    import time

    import joblib

    mlp = joblib.load('models/tmp_model.pkl')

    t0 = time.time()
    boards, moves, a_wins = auto_rematch(random_move, random_move, args_2=tuple({True}), args_1=tuple({True}),
                                         n_matches=100)

    print(a_wins)

    td = time.time()
    print(f'{((td - t0) // 60):.0f} min {((td - t0) % 60):.1f} sec')

    entries = save_data(boards, moves, filename='TMP.mat')

    print(entries)