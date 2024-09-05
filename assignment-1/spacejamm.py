from typing import List, Type, Tuple
import random
import pygame

import constants as c
import utils as u
from tiles import Tile, StartTile, GoalTile
from copy import deepcopy
from board import Board


class SpaceJamm:
    def __init__(self, filename: str) -> None:
        with open(filename) as file:
            lines = file.read().split("\n")
        self.board = Board.initialize_board(lines)

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

    def get_blitables(self):
        return self.board.get_blitables()

    def goal_test(self):
        return self.board.goal_test()
