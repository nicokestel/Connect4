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
        Name of second agent
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
    players = (PLAYER1, PLAYER2)
    board_player1 = []
    board_player2 = []
    moves1 = []
    moves2 = []
    num_boards_1 = 0
    num_boards_2 = 0
    num_moves_1 = 0
    num_moves_2 = 0
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
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {"X" if player == PLAYER1 else "O"}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                apply_player_action(board, action, player)

                # save board and player actions
                if player is PLAYER1:
                    moves1.append(action)
                    board_player1.append(board)
                    num_boards_1 += 1
                    num_moves_1 += 1
                else:
                    moves2.append(action)
                    board_player2.append(board)
                    num_boards_2 += 1
                    num_moves_2 += 1

                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        # TODO: what should be returned in case of a draw?
                        # return last player's moves if a draw occurs
                        if player is PLAYER1:
                            print("Game ended in draw")
                            print("n_boards_1: ", num_boards_1)
                            print("n_boards_2: ", num_boards_2)
                            print("n_moves_1: ", num_moves_1)
                            print("n_moves_2: ", num_moves_2)
                            return board_player1, moves1
                        else:
                            print("Game ended in draw")
                            print("n_boards_1: ", num_boards_1)
                            print("n_boards_2: ", num_boards_2)
                            print("n_moves_1: ", num_moves_1)
                            print("n_moves_2: ", num_moves_2)
                            return board_player1, moves2
                    else:
                        # game ended with 'player' as winner
                        print(
                            f'{player_name} won playing {"X" if player == PLAYER1 else "O"}'
                        )
                        if player is PLAYER1:
                            print("n_boards_1: ", num_boards_1)
                            print("n_boards_2: ", num_boards_2)
                            print("n_moves_1: ", num_moves_1)
                            print("n_moves_2: ", num_moves_2)
                            return board_player1, moves1
                        else:
                            print("n_boards_1: ", num_boards_1)
                            print("n_boards_2: ", num_boards_2)
                            print("n_moves_1: ", num_moves_1)
                            print("n_moves_2: ", num_moves_2)
                            return board_player2, moves2
                    playing = False
                    break
