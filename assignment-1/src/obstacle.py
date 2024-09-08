import pygame
from typing import List, Tuple

import atomics as a
import constants as c


class Obstacle(a.GameObject):
    obstacles_manufactured = 0

    def __init__(self, x: int, y: int, size: int, is_vertical: bool, uid=None) -> None:
        self.x = x
        self.y = y
        self.size = abs(size)
        self.is_vertical = is_vertical
        self.collided = False
        if uid == None:
            self.uid = Obstacle.obstacles_manufactured = (
                Obstacle.obstacles_manufactured + 1
            )
        else:
            self.uid = uid

        self.blit_coordinates = None
        self.sprite = None
        self.identity = None
        self.hitbox = set()

        self.build_sprite()
        self.create_identity()
        self.build_hitbox()

    def build_sprite(self):
        """Builds the sprite for each obstacle"""
        tile_size = c.TILE_SIZE
        factor = c.FACTOR
        sx, sy = tile_size, tile_size
        beam_surface = pygame.transform.scale(
            pygame.image.load("../assets/beam.png"), (tile_size // 4, c.PADDING * 4)
        )
        if not self.is_vertical:
            beam_surface = pygame.transform.rotate(beam_surface, 90)
        block_surface = pygame.transform.scale_by(
            pygame.image.load("../assets/block.png"), factor
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
        self.identity = (self.uid, self.x, self.y, self.size, self.is_vertical)

    def compress(self):
        return self.identity

    @staticmethod
    def decompress(data):
        return Obstacle(*data[1:], uid=data[0])

    def build_hitbox(self):
        for i in range(self.size):
            x = self.x + int(not self.is_vertical) * i
            y = self.y + int(self.is_vertical) * i
            self.hitbox.add((x, y))

    def build_hitbox_mapgen(self):
        for i in range(self.size):
            x = self.x + int(self.is_vertical) * i
            y = self.y + int(not self.is_vertical) * i
            self.hitbox.add((x, y))

    def get_possible_moves(self, board_size: int) -> List[Tuple[int, int]]:
        start = self.x * int(not self.is_vertical) + self.y * int(self.is_vertical)
        other = self.y * int(not self.is_vertical) + self.x * int(self.is_vertical)
        for curr_start in range(start, board_size - self.size + 1):
            if self.collided: break
            ret = (curr_start, other) if not self.is_vertical else (other, curr_start)
            yield ret
        self.collided = False

        for curr_start in range(start, -1, -1):
            if self.collided: break
            ret = (curr_start, other) if not self.is_vertical else (other, curr_start)
            yield ret

            #x = i * int(not self.is_vertical) + self.x * int(self.is_vertical)
            #y = i * int(self.is_vertical) + self.y * int(not self.is_vertical)
        self.collided = False

    def get_blitables(self):
        return self.sprite, self.blit_coordinates

    def __repr__(self) -> str:
        return f"Obstacle[{self.uid}] {self.identity} {self.hitbox}"
