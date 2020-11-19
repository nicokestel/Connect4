from enum import Enum
from typing import Optional
from typing import Callable, Tuple
import numpy as np

COLUMNS = np.int8(7)  # number of columns on board
ROWS = np.int(6)  # number of rows on board

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """

    board = np.tile(NO_PLAYER, (ROWS, COLUMNS)).astype(BoardPiece)
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    # add header
    board_str = '|' + COLUMNS * '==' + '|'

    for i in range(ROWS):
        # left border
        board_str += '\n|'
        for j in range(COLUMNS):
            # player pieces
            if board[ROWS - i - 1, j] == NO_PLAYER:
                board_str += '  '
            elif board[ROWS - i - 1, j] == PLAYER1:
                board_str += 'X '
            elif board[ROWS - i - 1, j] == PLAYER2:
                board_str += 'O '
        # right border
        board_str += '|'
    # bottom separator
    board_str += '\n|' + COLUMNS * '==' + '|'
    # footer
    board_str += '\n|'
    for j in range(COLUMNS):
        board_str += str(j) + ' '
    board_str += '|'
    return board_str


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    # board to return after board pieces are placed
    board = initialize_game_state()

    # remove header and footer
    rows = pp_board.splitlines()
    rows = rows[1:-2]

    # process row-wise
    for r in range(ROWS):
        row = rows[r].replace('|', '')
        for i in range(COLUMNS):
            piece = row[i * 2: (i * 2) + 1]
            if piece == 'O':
                board[COLUMNS - r - 2, i] = PLAYER2  # -2 correction because r is in 0..5 (not 1..6)
            elif piece == 'X':
                board[COLUMNS - r - 2, i] = PLAYER1

    return board


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """

    # optionally copy board
    if copy:
        mod_board = board.copy()
    else:
        mod_board = board

    # check desired column
    col = mod_board[:, action]
    # if column is not full
    if col[-1] == NO_PLAYER:
        mod_board[col.argmin(), action] = player

    return mod_board


def get_non_full_columns(board: np.ndarray) -> Tuple[PlayerAction]:
    """
    Receives a board and return the indices of playable columns.

    Parameters
    ----------
    board : np.ndarray
        Current board

    Returns
    -------
    actions : Tuple[PlayerAction]
        Playable moves on board

    """
    cpy_board = board.copy()
    cpy_board[cpy_board != NO_PLAYER] = PLAYER1

    actions = np.where(cpy_board.sum(axis=0) < 6)[0].astype(PlayerAction)
    return tuple(actions)


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    masked_board = (board == player).astype(BoardPiece)
    # print('\n', masked_board)

    if last_action:
        x0 = last_action
        y0 = board[:, last_action].argmin() - 1  # highest entry with piece

        # horizontal
        for j in range(COLUMNS - 3):
            if masked_board[y0, j] and masked_board[y0, j + 1] and masked_board[y0, j + 2] and masked_board[y0, j + 3]:
                return True

        # vertical
        for i in range(ROWS - 3):
            if masked_board[i, x0] and masked_board[i + 1, x0] and masked_board[i + 2, x0] and masked_board[i + 3, x0]:
                return True

        # check diagonals with count walker
        # walk along diagonals until non-player piece occurs
        directions = [(-1, -1), (-1, 1)]  # one for each diagonal
        for y_step, x_step in directions:
            count = 0
            xi = x0
            yi = y0
            while xi in range(0, COLUMNS) and yi in range(0, ROWS):
                if masked_board[yi, xi]:
                    yi += y_step
                    xi += x_step
                    count += 1
                else:
                    yi += y_step
                    xi += x_step
                    continue

                if count >= 4:
                    return True

            yi = y0 - y_step
            xi = x0 - x_step
            while xi in range(0, COLUMNS) and yi in range(0, ROWS):
                if masked_board[yi, xi]:
                    count += 1
                else:
                    break

                yi -= y_step
                xi -= x_step

                if count >= 4:
                    return True
        return False

    else:
        # horizontal
        for i in range(ROWS):
            for j in range(COLUMNS - 3):
                if masked_board[i, j] and masked_board[i, j + 1] and masked_board[i, j + 2] and masked_board[i, j + 3]:
                    return True

        # vertical
        for j in range(COLUMNS):
            for i in range(ROWS - 3):
                if masked_board[i, j] and masked_board[i + 1, j] and masked_board[i + 2, j] and masked_board[i + 3, j]:
                    return True

        # / diagonals
        for i in range(ROWS - 3):
            for j in range(3, COLUMNS):
                if masked_board[i, j] and masked_board[i + 1, j - 1] and masked_board[i + 2, j - 2] \
                        and masked_board[i + 3, j - 3]:
                    return True

        # \ diagonals
        for i in range(3, ROWS):
            for j in range(3, COLUMNS):
                if masked_board[i, j] and masked_board[i - 1, j - 1] and masked_board[i - 2, j - 2] \
                        and masked_board[i - 3, j - 3]:
                    return True

    return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """

    if connected_four(board, player, last_action):
        return GameState.IS_WIN

    # upper row needs to be full to be drawn
    if np.all(board[-1, :]):
        return GameState.IS_DRAW

    return GameState.STILL_PLAYING
