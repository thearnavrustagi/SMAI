from typing import List, Type
import numpy as np
from numpy.typing import NDArray
from enum import Enum

MAP_SIZE = 4
LOWER_BOUND = 0

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class Ship:
    @staticmethod
    def clamp(n):
        return max(LOWER_BOUND, min(n, MAP_SIZE))

    clamp_ = np.vectorize(clamp)
    
    def __init__(self, x: NDArray[np.int8], y: NDArray[np.int8], size: int, is_vertical: bool) -> None:
        self.x = x
        self.y = y
        self.size = size
        self.is_vertical = is_vertical
        # may need another variable to indicate the target spaceship
    

    def move(self, direction: Direction):
        # arbaaz the goat
        # do we check if legal move here or do we handle it later in the chain
        dx, dy = direction.value

        self.x += dx
        self.y += dy

        self.x = Ship.clamp_(self.x)
        self.y = Ship.clamp_(self.y)
        
        return
        
class Tile:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.occupied = False
        self.is_goal = False
    
    def set_goal(self):
        try:
            assert not self.occupied
            self.is_goal = True
        except AssertionError:
            # deal with this
            pass

    def set_occupied(self):
        try:
            assert not self.is_goal # Do we allow goal tile to be occupied
            self.occupied = True
        except AssertionError:
            # deal with this
            pass
            
    # having the set of occupied tiles would be easier to check than all of the ships coordinates?
    # generate Tiles then set occupied

class SpaceJamm:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = []
        for i in range(board_size):
            for j in range(board_size):
                self.board.append(Tile(i, j))
    
    # Set start tiles, goal tile, occupied tiles
    # keep order
    def set_board(self):
        pass

    # Define move gen here right
    def move_gen(self):
        pass

    def goal_test(self):
        pass
        
