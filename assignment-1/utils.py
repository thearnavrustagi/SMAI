import constants as c
from rich.console import Console

console = Console()


def clamp(n: int):
    return max(c.LOWER_BOUND, min(n, c.UPPER_BOUND))
