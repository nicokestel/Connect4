from agents.common import GenMove
from agents.common import PLAYER1, PLAYER2, GameState
from agents.common import initialize_game_state, apply_player_action, check_end_state
from typing import Tuple, List, Optional, Dict
import numpy as np


def evaluate(
        generate_move_1: GenMove,
        generate_move_2: GenMove,
        player_1: str = "Player 1",
        player_2: str = "Player 2",
        args_1: tuple = (),
        args_2: tuple = (),
        n_matches: np.int64 = 1000,
) -> Tuple[int, int, int]:
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
        Name of second agent
    args_1 : Tuple
        Arguments for generate_move_1
    args_2 : Tuple
        Arguments for generate_move_2
    n_matches : np.int64
        Number of matches to play (only games with one winner count)
        A match consists of two games with alternating first moving player

    Returns
    -------
    first_agent_wins : np.int32
        Wins of first agent
    second_agent_wins : np.int32
        Wins of second agent
    draws : np.int32
        Number of games ending in draw
    """

    # two games per match
    n_games = 2 * n_matches

    # return variables
    n_agent_wins = {PLAYER1: 0, PLAYER2: 0}
    n_draws = 0

    for n in range(n_matches):

        players = (PLAYER1, PLAYER2)
        for play_first in (1, -1):
            saved_state = {PLAYER1: None, PLAYER2: None}
            board = initialize_game_state()
            gen_moves = (generate_move_1, generate_move_2)[::play_first]
            player_names = (player_1, player_2)[::play_first]
            gen_args = (args_1, args_2)[::play_first]

            playing = True
            while playing:
                for player, player_name, gen_move, args in zip(
                    players, player_names, gen_moves, gen_args,
                ):
                    action, saved_state[player] = gen_move(
                        board.copy(), player, saved_state[player], *args
                    )

                    apply_player_action(board, action, player)

                    end_state = check_end_state(board, player)
                    if end_state != GameState.STILL_PLAYING:
                        if end_state == GameState.IS_DRAW:
                            # draw's do not count
                            # start new game
                            # reset boards and moves
                            n_draws += 1
                            board = initialize_game_state()
                            break
                        else:
                            # increment wins of winning agent
                            n_agent_wins[player] += 1
                        playing = False
                        break

    return n_agent_wins[PLAYER1], n_agent_wins[PLAYER2], n_draws


if __name__ == '__main__':
    from agents.agent_random import generate_move as random_move
    from agents.agent_mlp import generate_move as mlp_move
    import joblib

    mlp = joblib.load('agent_mlp/mlp_training/models/6_new_arch.pkl')

    f, s, d = evaluate(mlp_move, random_move, args_1=tuple({mlp}), args_2=tuple({True}), n_matches=1000)

    print(f, s, d)