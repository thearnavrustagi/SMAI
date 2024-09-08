from typing import List, Type, Tuple, Callable
import random
import pygame
from time import sleep
from collections import deque
from queue import PriorityQueue


import constants as c
import utils as u
from tiles import Tile, StartTile, GoalTile
from copy import deepcopy
from board import Board
from heuristics import jamm_heuristic

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
            sleep(0.3)

        sleep(5)
        return

    def depth_first_search(self):
        covered_tiles = set()
        stack = [self.board.compress()]
        trail = []
        stats = []
        level = 0

        while not self.goaltest():
            state = stack.pop()
            level += 1
            if state in covered_tiles:
                continue
            else:
                self.board = Board.decompress(state)
                children = list(self.movegen())
                u.track(stats, level, len(children))
                stack += children

            trail.append(state)
            covered_tiles.add(state)
        return trail, stats

    
    def breadth_first_search(self):
        covered_tiles = set()
        start = self.board.compress()
        queue = deque()
        trail = []
        stats = []
        parent = {start: None}
        state = start
        level = 0

        queue.append((start, level))

        while not self.goaltest():
            state, level = queue.popleft()
            level += 1
            if state in covered_tiles:
                continue
            self.board = Board.decompress(state)
            prior_state = len(queue)
            for child in self.movegen():
                if (child, level) not in covered_tiles:
                    parent[(child,level)] = state
                    queue.append((child, level))
            u.track(stats, level, len(queue) - prior_state)
            covered_tiles.add(state)

        while state:
            level -= 1
            trail.append(state)
            if not level: break
            state = parent[(state, level)]

        return trail[::-1], stats

    def best_first_search(self, heuristic:Callable=jamm_heuristic) -> List[Tuple[any]]:
        """
        Does best first search with a chosen heuristic
        """
        covered_tiles = set()
        start = self.board.compress()
        queue = PriorityQueue()
        trail = []
        parent = {start: None}
        state = start
        level = 0

        value = (start, level)
        queue.put((heuristic(*value), value))

        while not self.goaltest():
            _, (state, level) = queue.get()
            level += 1
            if state in covered_tiles:
                continue
            self.board = Board.decompress(state)
            for child in self.movegen():
                value = (child, level)
                if value not in covered_tiles:
                    parent[value] = state
                    queue.put((heuristic(*value), value))
            covered_tiles.add(state)

        while state:
            level -= 1
            trail.append(state)
            if not level: break
            state = parent[(state, level)]

        return trail[::-1]
