from agents.common import GenMove
from agents.common import PLAYER1, PLAYER2, GameState
from agents.common import initialize_game_state, apply_player_action, check_end_state
from typing import Tuple, List, Optional, Dict
import numpy as np


def auto_rematch(
        generate_move_1: GenMove,
        generate_move_2: GenMove,
        player_1: str = "Player 1",
        player_2: str = "Player 2",
        args_1: tuple = (),
        args_2: tuple = (),
        n_matches: np.int64 = 1000,
        sa_ratio: float = -1.0
) -> Tuple[List[np.ndarray], List[np.int8], Dict[int, int]]:
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
    sa_ratio : np.float32
        Ratio of games, that the second agent needs to win (measured on minimum number of games)

    Returns
    -------
    boards : List[np.ndarray]
        Boards of winner's moves
    moves : List[np.int8]
        Winner's moves
    n_wins : Dict[int, int]
        Number of won games for each agent
    """

    # two games per match
    n_games = 2 * n_matches

    # number of games the agents won
    a_wins = {PLAYER1: 0, PLAYER2: 0}
    a_wins_needed = {PLAYER1: (1-sa_ratio) * n_games, PLAYER2: sa_ratio * n_games}
    # a_wins_left = {PLAYER1: (1-sa_ratio) * n_games, PLAYER2: sa_ratio * n_games}

    # boards and moves that get returned after n_matches matches
    boards, moves = list(), list()
    # winning_boards = {PLAYER1: list(), PLAYER2: list()}
    # winning_moves = {PLAYER1: list(), PLAYER2: list()}

    n = 0
    while n < n_matches:
        if 0 <= sa_ratio <= 1:
            if a_wins[PLAYER1] == a_wins_needed[PLAYER1] and a_wins[PLAYER2] == a_wins_needed[PLAYER2]:
                break

        # print('starting match', n+1)
        players = (PLAYER1, PLAYER2)
        for play_first in (1, -1):
            saved_state = {PLAYER1: None, PLAYER2: None}
            board = initialize_game_state()
            gen_moves = (generate_move_1, generate_move_2)[::play_first]
            player_names = (player_1, player_2)[::play_first]
            gen_args = (args_1, args_2)[::play_first]

            # initialize tmp boards and moves lists
            tmp_boards = {PLAYER1: list(), PLAYER2: list()}
            tmp_moves = {PLAYER1: list(), PLAYER2: list()}

            playing = True
            while playing:
                for player, player_name, gen_move, args in zip(
                    players, player_names, gen_moves, gen_args,
                ):
                    action, saved_state[player] = gen_move(
                        board.copy(), player, saved_state[player], *args
                    )

                    # replace own pieces with 1 and opponent's pieces with -1
                    tmp_board = board.copy()
                    tmp_board[tmp_board == (PLAYER2 if player == PLAYER1 else PLAYER1)] = -1
                    tmp_board[tmp_board == player] = 1

                    # store tmp boards and moves
                    tmp_boards[player].append(tmp_board.flatten().astype(np.int8))
                    tmp_moves[player].append(action)

                    apply_player_action(board, action, player)

                    end_state = check_end_state(board, player)
                    if end_state != GameState.STILL_PLAYING:
                        if end_state == GameState.IS_DRAW:
                            # draw's do not count
                            # start new game
                            # reset boards and moves
                            board = initialize_game_state()
                            break
                        else:
                            # if valid second agent win ratio is given
                            if 0 <= sa_ratio <= 1:
                                # if agent still needs to win games
                                if a_wins[player] < a_wins_needed[player]:
                                    boards.extend(tmp_boards[player])
                                    moves.extend(tmp_moves[player])

                                    # increment wins the agent needs to achieve
                                    a_wins[player] += 1
                                else:
                                    n -= 1
                            else:
                                boards.extend(tmp_boards[player])
                                moves.extend(tmp_moves[player])
                                # increment wins the agent needs to achieve
                                a_wins[player] += 1

                        playing = False
                        break

        n += 1

    return boards, moves, a_wins
