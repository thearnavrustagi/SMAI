from typing import List, Type, Tuple
from enum import Enum
import random

import constants as c
import utils as u
from tiles import Tile, StartTile, GoalTile
from copy import deepcopy


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class Direction(ExtendedEnum):
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
        """MAY BE BROKEN, HAVE NOT TESTED YET"""
        """Moves the ship"""
        dx, dy = direction.value

        self.x += dx
        self.y += dy

        if dx >= 0 and dy >= 0:
            self.x = u.clamp(self.x + dx + self.size - 1)
            self.y = u.clamp(self.y + dy + self.size - 1)
        else:
            self.x = u.clamp(self.x + dx)
            self.y = u.clamp(self.y + dy)

        # self.x = u.clamp(self.x)
        # self.y = u.clamp(self.y)

    def occupy(self, board: List[List[Tile]], x=None, y=None) -> None:
        if x == None:
            x = self.x
        if y == None:
            y = self.y
        # x, y = self.x, self.y

        # print(x, y, self.size)
        for d in range(self.size):
            # print(x, y, d)
            if self.is_vertical:
                board[x][y + d].set_occupied(self.identity())
            else:
                board[x + d][y].set_occupied(self.identity())

    def unoccupy(self, board: List[List[Tile]], x=None, y=None) -> None:
        """Returns index of tile to be set to unoccupied after a move"""
        if x == None:
            x = self.x
        if y == None:
            y = self.y
        # x, y = self.x, self.y

        # print(x, y, self.size)
        for d in range(self.size):
            # print(x, y, d)
            if self.is_vertical:
                board[x][y + d].set_unoccupied()
            else:
                board[x + d][y].set_unoccupied()

    def valid_moves(self, board: List[List[Tile]]) -> List[List[List[Tile]]]:
        moves = []

        if self.is_vertical:
            direction = Direction.list()[:2]  # UP DOWN
        else:
            direction = Direction.list()[2:]  # LEFT RIGHT

        for move in direction:
            for factor in range(1, c.MAP_SIZE):
                x, y = move
                x *= factor
                y *= factor

                new_x = self.x + x
                new_y = self.y + y

                x_flag = int(not not x)
                y_flag = int(not not y)

                x_bounds = x_flag * (
                    new_x + self.size - 1
                )  # one of x_bounds and y_bounds is 0
                y_bounds = y_flag * (new_y + self.size - 1)

                if (
                    new_x < c.LOWER_BOUND
                    or new_y < c.LOWER_BOUND
                    or x_bounds > c.UPPER_BOUND
                    or y_bounds > c.UPPER_BOUND
                ):
                    continue

                collision = False
                for i in range(self.size):
                    x_idx, y_idx = new_x + x_flag * i, new_y + y_flag * i
                    if (
                        board[x_idx][y_idx].occupied
                        and self.identity() != board[x_idx][y_idx].occupied
                    ):
                        collision = True
                        break

                if not collision:
                    neighbour_state = deepcopy(board)
                    self.unoccupy(neighbour_state)
                    self.occupy(neighbour_state, new_x, new_y)

                    moves.append(neighbour_state)

        return moves

    def identity(self) -> Tuple[int, ...]:
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

    def initialize_ships(self, lines) -> None:
        for line in lines:
            if not line:
                continue
            (x, y, s) = tuple(int(a) for a in line.split(" "))
            ship = Ship(x, y, abs(s), s < 0)
            ship.occupy(self.board)
            self.ships.append(ship)

    @staticmethod
    def print_board(board) -> None:
        colors = {}
        for row in board:
            for tile in row:
                if isinstance(tile, StartTile):
                    u.console.print(c.START_SYMBOL, end="", style="bold green")
                elif isinstance(tile, GoalTile):
                    u.console.print(c.GOAL_SYMBOL, end="", style="bold red")
                else:
                    symbol, style = c.EMPTY_SYMBOL, "bold white"
                    if tile.occupied is not None:
                        symbol = c.SHIP_SYMBOL
                        if tile.occupied not in colors:
                            colors[
                                tile.occupied
                            ] = f"bold {random.choice(c.VALID_COLORS)}"
                        style = colors[tile.occupied]
                    u.console.print(
                        symbol,
                        style=style,
                        end="",
                    )
            u.console.print("")

    def show_board(self) -> None:
        self.print_board(self.board)

    # Define move gen here right
    def move_gen(self):
        states = []
        for ship in self.ships:
            moves = ship.valid_moves(self.board)
            if moves:
                u.console.print(
                    f"Showing all possible moves for ship [green]{ship.identity()}[/green]",
                    style="bold white",
                )
                for move in moves:
                    SpaceJamm.print_board(move)
                    print("=" * 80)
                    states.append(move)
        u.console.print(
            f"Number of neighbouring states = {len(states)}", style="bold green"
        )

    def goal_test(self):
        for tile in self.board[self.xs][1:-1]:
            if tile.occupied:
                return False
        print("GOAL STATE!!!")
        return True
