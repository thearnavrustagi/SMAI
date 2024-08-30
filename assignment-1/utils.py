import constants as c


def clamp(n: int):
    return max(c.LOWER_BOUND, min(n, c.MAP_SIZE))
