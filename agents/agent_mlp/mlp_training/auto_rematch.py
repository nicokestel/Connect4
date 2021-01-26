from agents.common import GenMove
from agents.common import PLAYER1, PLAYER2, GameState
from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state
from typing import Tuple, List
import numpy as np


def convert_arrays(boards: list, moves: list, player: int) -> (List, List):
    """
    Parameters
    ----------
    boards : list
        List of boards as numpy arrays
    moves: list
        List of moves as integers
    player: int
        Player from whose perspective the board is converted

    Returns
    -------
    converted: list
        List of boards with player pieces replaced with 1
        and opponent pieces replaced with -1
    converted_moves: list
        List of moves as np.int8 integers
    """
    res = []
    res_moves = []
    if player is PLAYER1:
        opponent = PLAYER2
    else:
        opponent = PLAYER1

    for board in boards:
        board = np.where(board == player, 1, board)
        board = np.where(board == opponent, -1, board)
        board = np.ndarray.flatten(board)
        board.astype('int8')
        res.append(board)
    for move in moves:
        res_moves.append(np.int8(move))
    return res, res_moves


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
    boards_player_1 = []
    boards_player_2 = []
    moves_player_1 = []
    moves_player_2 = []
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

                # save board and player actions
                if player is PLAYER1:
                    moves_player_1.append(action)
                    boards_player_1.append(board)
                else:
                    moves_player_2.append(action)
                    boards_player_2.append(board)

                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    if end_state == GameState.IS_DRAW:
                        # TODO: what should be returned in case of a draw?
                        # temporary solution: return last player's moves if a draw occurs
                        if player is PLAYER1:
                            boards, moves = convert_arrays(boards_player_1, moves_player_1, player)
                            return boards, moves
                        else:
                            boards, moves = convert_arrays(boards_player_2, moves_player_2, player)
                            return boards, moves
                    else:
                        # game ended with 'player' as winner
                        if player is PLAYER1:
                            boards, moves = convert_arrays(boards_player_1, moves_player_1, player)
                            return boards, moves
                        else:
                            boards, moves = convert_arrays(boards_player_2, moves_player_2, player)
                            return boards, moves
                    playing = False
                    break
