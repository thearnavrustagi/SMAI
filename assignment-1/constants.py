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

GAME_SIZE = (960, 960)
SPACE_BG = (8, 12, 26)
GRID_SIZE = 160
PADDING = 8
SPRITE_SIZE = 32

TILE_SIZE = GRID_SIZE - 2 * PADDING
FACTOR = TILE_SIZE // SPRITE_SIZE
