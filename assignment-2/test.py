def is_valid_tour(tour: list[int], num_cities: int) -> bool:
    """
    Check if the given tour is valid for a TSP.

    :param tour: The tour to check (a list of city indices)
    :param num_cities: Total number of cities (excluding return to start)
    :return: True if the tour is valid, False otherwise
    """
    # 1. Check if the tour starts and ends at the same city (typically city 0)
    if tour[0] != tour[-1]:
        print("Invalid: Tour does not start and end at the same city.")
        return False

    # 2. Check if the number of cities visited is correct
    # Tour should contain exactly num_cities + 1 cities (including the return to start)
    if len(tour) != num_cities + 1:
        print(
            f"Invalid: Tour has incorrect number of cities. Expected {num_cities + 1}, but got {len(tour)}."
        )
        return False

    # 3. Check if all cities are visited exactly once (except the start/end city)
    visited_cities = set(
        tour[:-1]
    )  # Exclude the last element (it's just the return to start)

    if len(visited_cities) != num_cities:
        print(
            f"Invalid: Not all cities are visited exactly once. Found {len(visited_cities)} unique cities."
        )
        return False

    # Check that city 0 is not visited more than twice (at start and end)
    if tour.count(0) != 2:
        print(
            f"Invalid: City 0 appears {tour.count(0)} times. It should appear exactly twice."
        )
        return False

    # Check that no other city appears more than once
    for city in range(1, num_cities):
        if tour.count(city) > 1:
            print(f"Invalid: City {city} appears more than once.")
            return False

    print("Valid tour.")
    return True


x = [
    0,
    5,
    18,
    14,
    11,
    52,
    84,
    69,
    59,
    78,
    96,
    6,
    30,
    58,
    74,
    46,
    57,
    70,
    3,
    21,
    56,
    37,
    76,
    99,
    17,
    24,
    31,
    82,
    54,
    7,
    19,
    28,
    66,
    44,
    13,
    60,
    34,
    73,
    90,
    35,
    98,
    68,
    26,
    29,
    51,
    16,
    10,
    95,
    71,
    39,
    47,
    38,
    25,
    49,
    20,
    87,
    55,
    23,
    36,
    22,
    83,
    41,
    97,
    63,
    89,
    33,
    72,
    1,
    27,
    9,
    81,
    93,
    91,
    43,
    50,
    12,
    75,
    32,
    64,
    79,
    62,
    42,
    85,
    86,
    65,
    2,
    77,
    53,
    15,
    88,
    40,
    48,
    67,
    92,
    8,
    80,
    94,
    61,
    45,
    4,
    0,
]

print(is_valid_tour(x, len(x) - 1))
