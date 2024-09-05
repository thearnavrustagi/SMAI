from typing import Tuple
import pygame

import constants as c


class Tile(object):
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.sprite = pygame.transform.scale_by(
            pygame.image.load("./assets/grid.png"), c.FACTOR
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.x}, {self.y}, {self.occupied})"

    def get_blitables(self):
        return self.sprite, (self.x * c.GRID_SIZE, self.y * c.GRID_SIZE)

    def compress(self):
        return (self.x, self.y)

    @staticmethod
    def decompress(data: Tuple[int, int]):
        return Tile(*data)


class GoalTile(Tile):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)
        self.sprite = pygame.transform.scale_by(
            pygame.image.load("./assets/goal.png"), c.FACTOR
        )

    @staticmethod
    def decompress(data: Tuple[int, int]):
        return GoalTile(*data)


class StartTile(Tile):
    def __init__(self, x: int, y: int, angle=0) -> None:
        super().__init__(x, y)
        self.sprite = pygame.transform.scale_by(
            pygame.image.load("./assets/ship.png"), c.FACTOR
        )
        self.sprite = pygame.transform.rotate(self.sprite, angle)

    @staticmethod
    def decompress(data: Tuple[int, int]):
        return StartTile(*data)
