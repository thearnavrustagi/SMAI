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
0, 180, 3, 127, 199, 89, 99, 171, 155, 36, 32, 61, 113, 56, 131, 133, 128, 124, 167, 161, 117, 181, 30, 85, 196, 134, 50, 8, 136, 25, 96, 37, 109, 73, 183, 55, 195, 86, 66, 138, 120, 84, 125, 95, 60, 75, 165, 64, 19, 159, 76, 157, 33, 106, 187, 119, 93, 59, 194, 51, 198, 137, 97, 118, 15, 11, 156, 72, 178, 174, 78, 68, 65, 5, 91, 170, 70, 88, 182, 176, 105, 12, 90, 142, 186, 166, 189, 49, 130, 110, 139, 163, 94, 74, 57, 42, 7, 28, 126, 62, 87, 115, 112, 101, 22, 81, 35, 193, 172, 107, 169, 132, 13, 111, 71, 148, 177, 44, 1, 147, 114, 10, 52, 46, 17, 151, 23, 162, 26, 43, 143, 83, 39, 191, 116, 63, 141, 4, 152, 168, 27, 24, 160, 175, 129, 6, 14, 149, 123, 108, 179, 67, 184, 29, 164, 69, 18, 80, 190, 34, 47, 100, 121, 53, 54, 98, 41, 31, 103, 58, 104, 158, 135, 38, 150, 40, 185, 145, 140, 45, 20, 197, 92, 21, 154, 173, 188, 16, 122, 102, 2, 82, 192, 153, 77, 79, 9, 144, 48, 146, 0
]

print(is_valid_tour(x, len(x) - 1))
