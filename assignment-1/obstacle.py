import pygame
from typing import List

import atomics as a
import constants as c


class Obstacle(a.GameObject):
    obstacles_manufactured = 0

    def __init__(self, x: int, y: int, size: int, is_vertical: bool) -> None:
        self.x = x
        self.y = y
        self.size = abs(size)
        self.is_vertical = is_vertical
        self.uid = Obstacle.obstacles_manufactured = Obstacle.obstacles_manufactured + 1

        self.blit_coordinates = None
        self.sprite = None
        self.identity = None

        self.build_sprite()
        self.create_identity()

    def build_sprite(self):
        """Builds the sprite for each obstacle"""
        tile_size = c.TILE_SIZE
        factor = c.FACTOR
        sx, sy = tile_size, tile_size
        beam_surface = pygame.transform.scale(
            pygame.image.load("./assets/beam.png"), (tile_size // 4, c.PADDING * 4)
        )
        if not self.is_vertical:
            beam_surface = pygame.transform.rotate(beam_surface, 90)
        block_surface = pygame.transform.scale_by(
            pygame.image.load("./assets/block.png"), factor
        )

        if self.is_vertical:
            sy = (tile_size + 1) * self.size + c.PADDING * 2 * (self.size - 1)
        else:
            sx = (tile_size + 1) * self.size + c.PADDING * 2 * (self.size - 1)

        self.sprite = pygame.Surface((sx, sy), pygame.SRCALPHA)
        self.blit_coordinates = (c.GRID_SIZE * self.x, c.GRID_SIZE * self.y)
        for i in range(self.size):
            if not self.is_vertical:
                location = (i * c.GRID_SIZE, 0)
                beam_location = (
                    c.GRID_SIZE * i - c.PADDING * 4,
                    c.PADDING + tile_size // 4,
                )
            else:
                location = (0, i * c.GRID_SIZE)
                beam_location = (
                    c.PADDING + tile_size // 4,
                    c.GRID_SIZE * i - 4 * c.PADDING,
                )
            self.sprite.blit(block_surface, location)
            if i < 1:
                continue
            self.sprite.blit(beam_surface, beam_location)

    def create_identity(self):
        wx = self.x + int(not self.is_vertical) * self.size
        wy = self.y + int(self.is_vertical) * self.size
        self.identity = (self.uid, self.x, self.y, wx, wy)

    """
    def valid_moves(self, board) -> List[List[List[Tile]]]:
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
        """

    def get_blitables(self):
        return self.sprite, self.blit_coordinates
