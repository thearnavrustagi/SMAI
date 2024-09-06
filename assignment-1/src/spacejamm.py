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

    def animate(self, trail):
        for state in trail:
            self.board = Board.decompress(state)
            self.blit_board()
            sleep(1)

        sleep(5)
        exit(0)

    def depth_first_search(self):
        covered_tiles = set()
        stack = [self.board.compress()]
        trail = []
        while not self.goaltest():
            state = stack.pop()
            if state in covered_tiles:
                continue
            else:
                self.board = Board.decompress(state)
                stack += list(self.movegen())

            trail.append(state)
            covered_tiles.add(state)
        return trail

    
    def breadth_first_search(self):
        covered_tiles = set()
        start = self.board.compress()
        queue = [start]
        trail = []
        parent = {start: None}
        state = start

        while not self.goaltest():
            state = queue.pop(0)
            if state in covered_tiles:
                continue
            else:
                self.board = Board.decompress(state)
                children = list(self.movegen())
                for child in children:
                    if child not in covered_tiles:
                        parent[child] = state

                queue += children

                covered_tiles.add(state)

        while state:
            trail.append(state)
            state = parent[state]

        return trail[::-1]
