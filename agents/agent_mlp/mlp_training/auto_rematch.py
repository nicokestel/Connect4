from agents.common import GenMove
from agents.common import PLAYER1, PLAYER2, GameState
from agents.common import initialize_game_state, apply_player_action, check_end_state
from typing import Tuple, List
import numpy as np


def auto_rematch(
        generate_move_1: GenMove,
        generate_move_2: GenMove,
        player_1: str = "Player 1",
        player_2: str = "Player 2",
        args_1: tuple = (),
        args_2: tuple = (),
        n_matches: np.int64 = 1000
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
    boards : List[np.ndarray]
        Boards of winner's moves
    moves : List[np.int8]
        Winner's moves
    """

    boards = list()
    moves = list()

    for n in range(n_matches):
        print('starting match', n+1)
        players = (PLAYER1, PLAYER2)
        for play_first in (1, -1):
            saved_state = {PLAYER1: None, PLAYER2: None}
            board = initialize_game_state()
            gen_moves = (generate_move_1, generate_move_2)[::play_first]
            player_names = (player_1, player_2)[::play_first]
            gen_args = (args_1, args_2)[::play_first]

            player_boards = {PLAYER1: list(), PLAYER2: list()}
            player_moves = {PLAYER1: list(), PLAYER2: list()}

            playing = True
            while playing:
                for player, player_name, gen_move, args in zip(
                    players, player_names, gen_moves, gen_args,
                ):
                    action, saved_state[player] = gen_move(
                        board.copy(), player, saved_state[player], *args
                    )

                    # store tmp boards and moves
                    tmp_board = board.copy()
                    tmp_board[tmp_board == (PLAYER2 if player == PLAYER1 else PLAYER1)] = -1
                    tmp_board[tmp_board == player] = 1
                    player_boards[player].append(tmp_board.flatten().astype(np.int8))
                    player_moves[player].append(action)

                    apply_player_action(board, action, player)

                    end_state = check_end_state(board, player)
                    if end_state != GameState.STILL_PLAYING:
                        if end_state == GameState.IS_DRAW:
                            # draw's do not count
                            # start new game
                            # reset boards and moves
                            player_boards = {PLAYER1: list(), PLAYER2: list()}
                            player_moves = {PLAYER1: list(), PLAYER2: list()}
                            board = initialize_game_state()
                            break
                        else:
                            # append boards to boards list
                            # append moves to moves list
                            boards.extend(player_boards[player])
                            moves.extend(player_moves[player])
                            pass
                        playing = False
                        break

    return boards, moves
