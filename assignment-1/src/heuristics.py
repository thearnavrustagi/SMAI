from typing import Tuple

from board import Board

def jamm_heuristic(state: Tuple[any], level: int) -> int:
    """
    Our primary heuristic, uses the number of 
    occupied grid squares summed with the depth
    """
    # board = Board.decompress(state)
    # num_blocked = board.get_blocked_tiles()
    num_blocked = state[-1]
    return num_blocked + level
