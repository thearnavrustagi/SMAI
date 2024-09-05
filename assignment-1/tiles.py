from typing import Tuple
import pygame

import constants as c


class Tile(object):
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.occupied = None
        self.sprite = pygame.transform.scale_by(
            pygame.image.load("./assets/grid.png"), c.FACTOR
        )

    def set_occupied(self, ship_identity: Tuple[int, ...]):
        self.occupied = ship_identity

    def set_unoccupied(self):
        self.occupied = None

    def __repr__(self):
        return f"Tile({self.x}, {self.y}, {self.occupied})"

    def get_blitables(self):
        return self.sprite, (self.x * c.GRID_SIZE, self.y * c.GRID_SIZE)


class GoalTile(Tile):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)
        tile_asset = pygame.transform.scale_by(
            pygame.image.load("./assets/grid.png"), c.FACTOR
        )
        main_asset = pygame.transform.scale_by(
            pygame.image.load("./assets/goal.png"), c.FACTOR
        )
        self.sprite = pygame.Surface((c.GRID_SIZE, c.GRID_SIZE), pygame.SRCALPHA)
        self.sprite.blit(tile_asset, (0, 0))
        self.sprite.blit(main_asset, (0, 0))


class StartTile(Tile):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)
        self.set_occupied(c.START_OCCUPANCY_IDENTITY)
        tile_asset = pygame.transform.scale_by(
            pygame.image.load("./assets/grid.png"), c.FACTOR
        )
        main_asset = pygame.transform.scale_by(
            pygame.image.load("./assets/ship.png"), c.FACTOR
        )
        self.sprite = pygame.Surface((c.GRID_SIZE, c.GRID_SIZE), pygame.SRCALPHA)
        self.sprite.blit(tile_asset, (0, 0))
        self.sprite.blit(main_asset, (0, 0))
