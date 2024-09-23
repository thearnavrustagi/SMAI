import sys
from typing import List, Tuple
from enum import Enum
import random

class GraphType(Enum):
    EUCLIDEAN = "EUCLIDEAN"
    NON_EUCLIDEAN = "NON_EUCLIDEAN"

class Graph:
    def __init__(self, graph_type: GraphType, num_cities: int):
        """
        Initialize the Graph object.

        :param graph_type: Type of the graph (EUCLIDEAN or NON_EUCLIDEAN)
        :param num_cities: Number of cities in the graph
        """
        self.graph_type = graph_type
        self.num_cities = num_cities
        self.coordinates: List[Tuple[float, float]] = []
        self.distances: List[List[float]] = []

    def add_coordinate(self, x: float, y: float) -> None:
        """
        Add a coordinate to the graph.

        :param x: X-coordinate
        :param y: Y-coordinate
        """
        self.coordinates.append((x, y))

    def add_distance_row(self, distances: List[float]) -> None:
        """
        Add a row of distances to the graph.

        :param distances: List of distances to other cities
        """
        self.distances.append(distances)

    def calculate_tour_cost(self, tour: List[int]) -> float:
        """
        Calculate the cost of a given tour.

        :param tour: List of city indices representing the tour
        :return: Total cost of the tour
        """
        return sum(self.distances[tour[i]][tour[(i + 1) % self.num_cities]]
                   for i in range(self.num_cities))

    def generate_random_tour(self) -> List[int]:
        """
        Generate a random tour.

        :return: List of city indices representing a random tour
        """
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour

    def kruskal_mst():
        parents = []

    def eulerian_cycle():
        pass


def read_input() -> Graph:
    """
    Read input from standard input and create a Graph object.

    :return: Initialized Graph object
    """
    graph_type = GraphType(input().strip())
    num_cities = int(input().strip())
    graph = Graph(graph_type, num_cities)

    for _ in range(num_cities):
        x, y = map(float, input().strip().split())
        graph.add_coordinate(x, y)

    for _ in range(num_cities):
        distances = list(map(float, input().strip().split()))
        graph.add_distance_row(distances)

    return graph

def solve_tsp(graph: Graph) -> None:
    """
    Solve the TSP problem using a simple random search algorithm.

    :param graph: Graph object representing the TSP instance
    """
    best_tour = graph.generate_random_tour()
    best_cost = graph.calculate_tour_cost(best_tour)

    for _ in range(1000):  # Perform 1000 iterations
        current_tour = graph.christofides()
        current_cost = graph.calculate_tour_cost(current_tour)

        if current_cost < best_cost:
            best_tour = current_tour
            best_cost = current_cost
            print(" ".join(map(str, best_tour)))

def main() -> None:
    """
    Main function to read input, solve TSP, and output results.
    """
    graph = read_input()
    solve_tsp(graph)

if __name__ == "__main__":
    main()
