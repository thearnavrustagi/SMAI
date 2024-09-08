from typing import Tuple, List
from numpy.random import randint, normal
from itertools import groupby
import random

import constants as c
from rich.console import Console
from obstacle import Obstacle

console = Console()


def clamp(n: int, lower_bound, upper_bound):
    return int(max(lower_bound, min(n, upper_bound)))


def collisions(obstacle_set: set[Tuple[int, int]], obstacles: List[Obstacle]) -> bool:
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

def valid_obstacle(available, ind, s):
    keys = []
    coords = list(available)
    coords.sort()
    func = lambda item: item[ind]
    for key, group in groupby(coords, func):
        group_list = list(group)
        if len(group_list) >= s:
            max_ind = max(group_list, key=lambda item: item[not ind])[not ind]
            min_ind = min(group_list, key=lambda item: item[not ind])[not ind]
            keys.append((key, random.choice(range(min_ind, max_ind))))
    
    if len(keys) == 0:
        return -1, -1
    return random.choice(keys)
    
def obstacle_gen(map_size, n_obs, start_row, board):
    MAX_HEIGHT = map_size - start_row + 1
    available= set((i, j) for i in range(map_size) for j in range(map_size))
    available -= board.obstacles[0].hitbox
    while n_obs:
        # print(n_obs)
        s = randint(2, MAX_HEIGHT)
        if not randint(3):
            s = -s
        (x, y) = random.choice(list(available))
            
        temp = Obstacle(x, y, abs(s), s < 0)
        temp.build_hitbox_mapgen()
        if temp.hitbox <= available:
            board.obstacles.append(temp)
            available -= temp.hitbox
            n_obs -= 1
    return board.obstacles
