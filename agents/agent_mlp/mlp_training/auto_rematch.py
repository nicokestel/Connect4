from agents.common import GenMove
from agents.common import PLAYER1, PLAYER2, GameState
from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state
from typing import Tuple, List
import numpy as np


def auto_rematch(
        generate_move_1: GenMove,
        generate_move_2: GenMove,
        player_1: str = "Player 1",
        player_2: str = "Player 2",
        args_1: tuple = (),
        args_2: tuple = ()
) -> Tuple[List[np.ndarray], List[np.int8]]:
    """

    Parameters
    ----------
    generate_move_1 : GenMove
        Strategy of first agent
    generate_move_2 : GenMove
        Strategy of second agent
    player_1 : str
        Name of first agent
    player_2 : str
        Name of seconde agent
    args_1 : Tuple
        Arguments for generate_move_1
    args_2 : Tuple
        Arguments for generate_move_2

    Returns
    -------
    boards : List[np.ndarray]
        Boards of winner's moves
    moves : List[np.int8]
        Winner's moves
    """
    pass