from typing import Tuple
import constants as c


class Tile(object):
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.occupied = None

    def set_occupied(self, ship_identity: Tuple[int, ...]):
        self.occupied = ship_identity

    def set_unoccupied(self):
        self.occupied = None

    def __repr__(self):
        return f"Tile({self.x}, {self.y}, {self.occupied})"


class GoalTile(Tile):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)


class StartTile(Tile):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)
        self.set_occupied(c.START_OCCUPANCY_IDENTITY)
