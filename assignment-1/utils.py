from typing import Tuple, List

import constants as c
from rich.console import Console
from obstacle import Obstacle

console = Console()


def clamp(n: int):
    return max(c.LOWER_BOUND, min(n, c.UPPER_BOUND))


def collisions(obstacle_set: Tuple[int, int], obstacles: List[Obstacle]) -> bool:
    for obstacle in obstacles:
        if len(obstacle_set & obstacle.hitbox):
            return True
    return False


def insert_state(compressed, obstacle, new_move):
    idx = compressed.index(obstacle)
    obstacle = list(obstacle)
    obstacle[1], obstacle[2] = new_move
    compressed = list(compressed)
    compressed[idx] = tuple(obstacle)
    return tuple(compressed)
