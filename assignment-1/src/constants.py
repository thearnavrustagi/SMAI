import atomics as a


class Direction(a.ExtendedEnum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


MAP_SIZE = 4

LOWER_BOUND = 0
UPPER_BOUND = MAP_SIZE - 1

START_OCCUPANCY_IDENTITY = (-1, -1, -1, -1)

START_SYMBOL = "^"
GOAL_SYMBOL = "_"
SHIP_SYMBOL = "#"
EMPTY_SYMBOL = "."

VALID_COLORS = (
    "magenta blue cyan white yellow bright_blue bright_yellow bright_magenta".split()
)

GAME_SIZE = (1080, 1080)
SPACE_BG = (8, 12, 26)
GRID_SIZE = 124
PADDING = 8
SPRITE_SIZE = 32

TILE_SIZE = GRID_SIZE - 2 * PADDING
FACTOR = TILE_SIZE // SPRITE_SIZE

BLOCK_ASSETS = ["../assets/block-1.png", "../assets/block-2.png", "../assets/block-3.png"]
