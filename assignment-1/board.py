from typing import List
import pygame
from math import atan2, pi

import atomics as a
import constants as c
from tiles import StartTile, GoalTile, Tile
from obstacle import Obstacle


class Board(a.GameObject):
    def __new__(cls):
        """Make Board a singleton class"""
        if not hasattr(cls, "instance"):
            cls.instance = super(Board, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        """
        self.tile_sprite -> tile sprite to be used for the board
        self.occupied -> set of occupied tiles, Tuple of (x, y) -> Obstacle object
        """
        self.tile_sprite = pygame.transform.scale_by(
            pygame.image.load("./assets/grid.png"), c.FACTOR
        )
        self.occupied = {}

    @staticmethod
    def initialize_board(lines):
        board = Board()
        board.size = int(lines[0])
        board.obstacles = []

        board.initialize_primary_tiles(lines[1], lines[2])
        board.initialize_obstacles(lines[3:])

        return board

    def initialize_primary_tiles(self, start, end):
        xg, yg = tuple(int(a) for a in end.split(" "))
        self.goal_tile = GoalTile(xg, yg)

        xs, ys = tuple(int(a) for a in start.split(" "))
        angle = int(xg > xs) * -90 + int(xg < xs) * 90
        angle += int(yg > ys) * 180 + int(yg < ys) * 0

        self.start_tile = StartTile(xs, ys, angle)

    def initialize_obstacles(self, lines) -> None:
        self.obstacles = []

        for line in lines:
            if not line:
                continue
            (x, y, s) = tuple(int(a) for a in line.split(" "))
            obstacle = Obstacle(x, y, abs(s), s < 0)
            self.obstacles.append(obstacle)

    def get_blitables(self):
        for x in range(self.size):
            for y in range(self.size):
                yield self.tile_sprite, (x * c.GRID_SIZE, y * c.GRID_SIZE)
        yield self.start_tile.get_blitables()
        yield self.goal_tile.get_blitables()
        for obstacle in self.obstacles:
            yield obstacle.get_blitables()

    def goal_test(self):
        if self.start_tile.x == self.goal_tile.x:
            x = self.start_tile.x
            for y in range(x, self.goal_tile.y):
                if (x, y) in self.occupied:
                    return False
        elif self.start_tile.y == self.goal_tile.y:
            y = self.start_tile.y
            for x in range(x, self.goal_tile.x):
                if (x, y) in self.occupied:
                    return False
        else:
            return False
        return True
