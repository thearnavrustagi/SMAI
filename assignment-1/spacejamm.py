from typing import List, Type, Tuple
import random
import pygame
from time import sleep

import constants as c
import utils as u
from tiles import Tile, StartTile, GoalTile
from copy import deepcopy
from board import Board


class SpaceJamm:
    def __init__(self, filename: str, screen) -> None:
        with open(filename) as file:
            lines = file.read().split("\n")
        self.board = Board.initialize_board(lines)
        self.screen = screen

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

    def blit_board(self):
        self.screen.fill(c.SPACE_BG)
        for blitable in self.get_blitables():
            self.screen.blit(*blitable)
        pygame.display.flip()

    def get_blitables(self):
        return self.board.get_blitables()

    def goaltest(self):
        return self.board.goaltest()

    def movegen(self):
        for move in self.board.movegen():
            yield move

    def depth_first_search(self, covered_tiles=None):
        covered_tiles = set() if covered_tiles is None else covered_tiles
        valid_states = list(self.movegen())
        for state in valid_states:
            if state in covered_tiles:
                continue
            if self.goaltest():
                sleep(15)
                exit(0)
            self.board = Board.decompress(state)
            covered_tiles.add(state)
            sleep(1)
            self.blit_board()
            self.depth_first_search(covered_tiles)
