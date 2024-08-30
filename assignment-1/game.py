from typing import List, Type, Tuple
import numpy as np
from numpy.typing import NDArray
from enum import Enum

import constants as c
import utils as u
from tiles import Tile, StartTile, GoalTile


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Ship:
    def __init__(self, x: int, y: int, size: int, is_vertical: bool) -> None:
        self.x = x
        self.y = y
        self.size = abs(size)
        self.is_vertical = is_vertical
        # may need another variable to indicate the target spaceship

    def move(self, direction: Direction):
        """Moves the ship"""
        dx, dy = direction.value

        self.x += dx
        self.y += dy

        self.x = u.clamp(self.x)
        self.y = u.clamp(self.y)

    def occupy(self, board: List[List[Tile]]) -> None:
        x, y = self.x, self.y
        print(x, y, self.size)
        for d in range(self.size):
            print(x, y, d)
            if self.is_vertical:
                board[x][y + d].set_occupied(self.identity())
            else:
                board[x + d][y].set_occupied(self.identity())

    def identity(self) -> Tuple[int]:
        return (self.x, self.y, self.size, int(self.is_vertical))


class SpaceJamm:
    def __init__(self, filename: str) -> None:
        with open(filename) as file:
            lines = file.read().split("\n")
        self.board_size = int(lines[0])
        self.initialize_board()
        self.initialize_tiles(lines[1:])

    def initialize_board(self) -> None:
        self.board = []
        self.ships = []

        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                row.append(Tile(i, j))
            self.board.append(row)

    def initialize_tiles(self, lines: List[str]) -> None:
        """Initializes the board with ships, goal & start tiles"""
        self.initialize_primary_tiles(lines[0], lines[1])
        self.initialize_ships(lines[2:])

    def initialize_primary_tiles(self, start, end):
        (self.xs, self.ys) = tuple(int(a) for a in start.split(" "))
        self.board[self.xs][self.ys] = StartTile(self.xs, self.ys)

        (self.xg, self.yg) = tuple(int(a) for a in end.split(" "))
        self.board[self.xg][self.yg] = GoalTile(self.xg, self.yg)

    def initialize_ships(self, lines):
        for line in lines:
            if not line:
                continue
            (x, y, s) = tuple(int(a) for a in line.split(" "))
            ship = Ship(x, y, abs(s), s < 0)
            ship.occupy(self.board)
            self.ships.append(ship)
            print("=" * 80)
            for row in self.board:
                print(row)
            print("=" * 80)

    def show_board(self) -> None:
        for row in self.board:
            for tile in row:
                if isinstance(tile, StartTile):
                    print(c.START_SYMBOL, end="")
                elif isinstance(tile, GoalTile):
                    print(c.GOAL_SYMBOL, end="")
                else:
                    print(
                        c.SHIP_SYMBOL if tile.occupied is not None else c.EMPTY_SYMBOL,
                        end="",
                    )
            print("")

    # Define move gen here right
    def move_gen(self):
        pass

    def goal_test(self):
        for tile in self.board[self.xs][1:-1]:
            if tile.occupied:
                return False
        return True
