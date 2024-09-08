from typing import List
import pygame
from math import atan2, pi

from numpy.random import randint, normal
import atomics as a
import constants as c
import utils as u
from tiles import StartTile, GoalTile, Tile
from obstacle import Obstacle


class Board(a.GameObject):
    def __init__(self):
        """
        self.tile_sprite -> tile sprite to be used for the board
        self.occupied -> set of occupied tiles, Tuple of (x, y) -> Obstacle object
        """
        self.tile_sprite = pygame.transform.scale_by(
            pygame.image.load("../assets/grid.png"), c.FACTOR
        )

        self.size = -1
        self.start_tile, self.goal_tile = None, None
        self.obstacles = []

    @staticmethod
    def mapgen(map_size, n_obs):
        board = Board()
        board.size = map_size
        
        start_row = (map_size + 1) // 2 - 1
        board.start_tile = StartTile(0, start_row)
        board.goal_tile = GoalTile(map_size-1, start_row)

        board.obstacles = []
        board.obstacles.append(Obstacle(0, start_row, map_size, True))
        board.obstacles[0].build_hitbox_mapgen()
        

        board.obstacles = u.obstacle_gen(map_size, n_obs, start_row, board)
        board.obstacles = board.obstacles[1:]
        return board

    @staticmethod
    def initialize_board(lines):
        board = Board()
        board.size = int(lines[0])
        board.obstacles = []

        board.initialize_primary_tiles(lines[1], lines[2])
        board.initialize_obstacles(lines[3:])

        return board

    @staticmethod
    def dump(board: "Board"):
        out = ""
        out += f"{board.size}\n"
        out += f"{board.start_tile.x} {board.start_tile.y}\n"
        out += f"{board.goal_tile.x} {board.goal_tile.y}\n"
        for obstacle in board.obstacles:
            is_vert = 1 if obstacle.is_vertical else -1
            out += f"{obstacle.x} {obstacle.y} {is_vert * obstacle.size}\n"

        return out
            

    def initialize_primary_tiles(self, start, end):
        xg, yg = tuple(int(a) for a in end.split(" ")) if isinstance(end, str) else end
        self.goal_tile = GoalTile(xg, yg)

        xs, ys = (
            tuple(int(a) for a in start.split(" ")) if isinstance(start, str) else start
        )
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

    def compress(self):
        compressed = [self.size, self.start_tile.compress(), self.goal_tile.compress()]
        for obstacle in self.obstacles:
            compressed.append(obstacle.compress())
        return tuple(compressed)

    @staticmethod
    def decompress(data):
        board = Board()
        board.size = data[0]
        board.initialize_primary_tiles(data[1], data[2])
        board.obstacles = []

        for obstacle in data[3:]:
            board.obstacles.append(Obstacle.decompress(obstacle))

        return board

    def get_blocked_tiles(self):
        tiles = None
        if self.start_tile.x == self.goal_tile.x:
            x = self.start_tile.x
            Y1, Y2 = self.start_tile.y, self.goal_tile.y + 1
            tiles = set(((x,y) for y in range(Y1, Y2, 1 if Y2 > Y1 else -1)))
        elif self.start_tile.y == self.goal_tile.y:
            y = self.start_tile.y
            X1, X2 = self.start_tile.x, self.goal_tile.x + 1
            tiles = set(((x,y) for x in range(X1, X2, 1 if X2 > X1 else -1)))
        else:
            return -1
        return u.sum_commons(tiles, self.obstacles)

    def get_blitables(self):
        for x in range(self.size):
            for y in range(self.size):
                yield self.tile_sprite, (x * c.GRID_SIZE, y * c.GRID_SIZE)
        yield self.start_tile.get_blitables()
        yield self.goal_tile.get_blitables()
        for obstacle in self.obstacles:
            yield obstacle.get_blitables()

    def goaltest(self):
        moves = set()
        if self.start_tile.x == self.goal_tile.x:
            x = self.start_tile.x
            Y1, Y2 = self.start_tile.y, self.goal_tile.y + 1
            for y in range(Y1, Y2, 1 if Y2 > Y1 else -1):
                moves.add((x, y))
        elif self.start_tile.y == self.goal_tile.y:
            y = self.start_tile.y
            X1, X2 = self.start_tile.x, self.goal_tile.x + 1
            for x in range(X1, X2, 1 if X2 > X1 else -1):
                moves.add((x, y))
        else:
            return False
        return not u.collisions(moves, self.obstacles)

    def movegen(self):
        compressed = self.compress()
        collided = False
        for i, obstacle in enumerate(self.obstacles):
            obstacle.collided = False
            for move in obstacle.get_possible_moves(self.size):
                obstacle.collided = False
                if move == (obstacle.x, obstacle.y):
                    continue
                obstacle_set = {
                    (
                        move[0] + i * int(not obstacle.is_vertical),
                        move[1] + i * int(obstacle.is_vertical),
                    )
                    for i in range(obstacle.size)
                }
                if self.start_tile.compress() in obstacle_set:
                    continue
                if self.goal_tile.compress() in obstacle_set:
                    continue
                obstacle.collided = u.collisions(
                    obstacle_set, self.obstacles[:i] + self.obstacles[i + 1 :]
                )
                if obstacle.collided:
                    continue
                state = u.insert_state(compressed, obstacle.compress(), move)
                yield state
