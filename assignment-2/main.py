import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import random
import copy
import sys
from typing import List, Tuple, Dict
from enum import Enum
import random
import math
import itertools

random.seed(0)


def find_parent(parent, i):
    if parent[i] != i:
        parent[i] = find_parent(parent, parent[i])
    return parent[i]


def union_find(parent, rank, x, y):
    """
    Attach a lower rank tree under the root of a higher rank tree

    :param parent:
    """
    if rank[x] < rank[y]:
        parent[x] = y
    elif rank[x] > rank[y]:
        parent[y] = x
    else:
        parent[y] = x
        rank[x] += 1


class GraphType(Enum):
    EUCLIDEAN = "EUCLIDEAN"
    NON_EUCLIDEAN = "NON-EUCLIDEAN"


class Graph:
    def __init__(self, graph_type: GraphType, num_cities: int):
        """
        Initialize the Graph object.

        :param graph_type: Type of the graph (EUCLIDEAN or NON_EUCLIDEAN)
        :param num_cities: Number of cities in the graph
        """
        self.graph_type = graph_type
        self.num_cities = num_cities
        self.coordinates = {}
        self.weights = {}

    def add_coordinate(self, u: int, coords) -> None:
        self.coordinates[u] = coords

    def add_weight(self, u: int, weights: List[float]) -> None:
        for v, w in enumerate(weights):
            if w == v:
                continue
            self.weights[(u, v)] = w

    def kruskal_mst(self):
        mst = []
        parents, rank = list(range(self.num_cities)), [0] * self.num_cities

        itr = 0
        edges = 0

        graph = [(*uv, w) for uv, w in self.weights.items()]
        graph = sorted(graph, key=lambda x: x[2])

        while edges < self.num_cities - 1:
            u, v, w = graph[itr]
            itr += 1
            x = find_parent(parents, u)
            y = find_parent(parents, v)

            if x != y:
                edges += 1
                mst.append([u, v, w])
                union_find(parents, rank, x, y)
        return mst

    def christofides(self):
        # Step 1: Create a minimum spanning tree
        mst = self.kruskal_mst()

        # Step 2: Find odd-degree vertices
        degree = [0] * self.num_cities
        for edge in mst:
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        odd_vertices = [i for i in range(self.num_cities) if degree[i] % 2 != 0]

        # Step 3: Find minimum-weight perfect matching (greedy approach)
        matching = self.greedy_matching(odd_vertices)

        # Step 4: Combine MST and matching
        eulerian_graph = mst + matching

        # Step 5: Find Eulerian circuit
        euler_circuit = self.find_eulerian_circuit(eulerian_graph)

        # Step 6: Make Hamiltonian circuit
        hamiltonian_circuit = self.make_hamiltonian(euler_circuit)

        return hamiltonian_circuit

    def greedy_matching(self, odd_vertices):
        matching = []
        unmatched = odd_vertices.copy()

        while unmatched:
            v = unmatched.pop(0)
            closest = min(
                unmatched, key=lambda u: self.weights.get((v, u), float("inf"))
            )
            matching.append([v, closest, self.weights.get((v, closest))])
            unmatched.remove(closest)

        return matching

    def get_optimal_tour(self, s):
        """
        Find the optimal tour using the Held-Karp algorithm (dynamic programming).

        :return: Tuple containing the optimal tour and its cost
        """
        n = self.num_cities

        # Initialize the DP table
        dp = {}

        # Base case: start from city 0 to each other city
        for i in range(1, n):
            dp[(1 << i, i)] = (self.weights.get((0, i), float("inf")), [0, i])

        # Iterate over all subsets of cities
        for size in range(2, n):
            for subset in itertools.combinations(range(1, n), size):
                bits = 0
                for bit in subset:
                    bits |= 1 << bit

                for last in subset:
                    prev = bits & ~(1 << last)
                    res = []
                    for j in subset:
                        if j == last:
                            continue
                        res.append(
                            (
                                dp[(prev, j)][0]
                                + self.weights.get((j, last), float("inf")),
                                j,
                            )
                        )
                    dp[(bits, last)] = min(res)

        # Find the optimal tour
        bits = (2**n - 1) - 1  # All bits set except the 0th city
        res = []
        for i in range(1, n):
            res.append((dp[(bits, i)][0] + self.weights.get((i, 0), float("inf")), i))
        opt, parent = min(res)

        # Reconstruct the path
        path = [0]
        for i in range(n - 1):
            path.append(parent)
            new_bits = bits & ~(1 << parent)
            _, parent = dp[(bits, parent)]
            bits = new_bits
        path.append(0)

        return path

    def find_eulerian_circuit(self, graph):
        # Create an adjacency list
        adj_list = {i: [] for i in range(self.num_cities)}
        for edge in graph:
            adj_list[edge[0]].append(edge[1])
            adj_list[edge[1]].append(edge[0])

        # Find Eulerian circuit
        circuit = []
        stack = [0]  # Start from vertex 0
        while stack:
            v = stack[-1]
            if adj_list[v]:
                u = adj_list[v].pop()
                stack.append(u)
                adj_list[u].remove(v)
            else:
                circuit.append(stack.pop())

        return circuit[::-1]

    def make_hamiltonian(self, circuit):
        visited = set()
        hamiltonian = []
        for v in circuit:
            if v not in visited:
                visited.add(v)
                hamiltonian.append(v)
        hamiltonian.append(hamiltonian[0])  # Return to start
        return hamiltonian

    def calculate_tour_cost(self, tour):
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.weights.get((tour[i], tour[i + 1]), float("inf"))
        return cost

    def distance(self, city1, city2):
        if city1 == city2:
            return 0
        return self.weights[(city1, city2)]

    def reverse_segment(self, tour, start, end):
        N = len(tour)
        size = ((N + end - start + 1) % N) // 2

        for _ in range(size):
            tour[start], tour[end] = tour[end], tour[start]
            start = (start + 1) % N
            end = (end - 1) % N

    def two_opt(self, tour):
        N = len(tour)
        locally_optimal = False

        while not locally_optimal:
            locally_optimal = True
            for i in range(N - 2):
                X1 = tour[i]
                X2 = tour[(i + 1) % N]

                counter_2_limit = N - 2 if i == 0 else N - 1

                for j in range(i + 2, counter_2_limit + 1):
                    Y1 = tour[j]
                    Y2 = tour[(j + 1) % N]

                    if self.two_opt_gain(X1, X2, Y1, Y2) > 0:
                        # this move will shorten the tour, apply it at once
                        self.reverse_segment(tour, (i + 1) % N, j)
                        locally_optimal = False
                        break

                if not locally_optimal:
                    break
        return tour

    def two_opt_gain(self, city1, city2, city3, city4):
        current = self.distance(city1, city2) + self.distance(city3, city4)
        swapped = self.distance(city1, city3) + self.distance(city2, city4)

        return current - swapped

    def lin_kernighan(self, initial_tour):
        """
        Implement the Lin-Kernighan heuristic for TSP.

        :param initial_tour: List representing the initial tour
        :return: Improved tour
        """
        best_tour = initial_tour.copy()
        best_cost = self.calculate_tour_cost(best_tour)

        def edge_cost(u, v):
            return self.weights.get((u, v), float("inf"))

        for v in range(self.num_cities):
            for edge in [best_tour[v - 1], best_tour[(v + 1) % self.num_cities]]:
                # Step 1: Outer loop for each node/edge pair
                t0 = best_tour.index(v)
                u0 = edge

                # Step 2: Initialize edge scan
                for w0 in range(self.num_cities):
                    if w0 != v and w0 != u0 and edge_cost(v, w0) < edge_cost(v, u0):
                        delta_path = [v, w0]
                        i = 0

                        while True:
                            # Step 3: Test tour
                            new_tour = self.construct_tour(best_tour, delta_path)
                            new_cost = self.calculate_tour_cost(new_tour)
                            if new_cost < best_cost:
                                best_tour = new_tour
                                best_cost = new_cost

                            # Step 4: Build next δ-path
                            ui = delta_path[-2]
                            wi = delta_path[-1]
                            ui_next = best_tour[
                                (best_tour.index(wi) + 1) % self.num_cities
                            ]

                            if ui_next in delta_path:
                                break  # Stop this scan

                            for wi_next in range(self.num_cities):
                                if wi_next not in delta_path and wi_next != ui_next:
                                    if edge_cost(ui_next, wi_next) < edge_cost(
                                        ui_next, wi
                                    ):
                                        delta_path.extend([ui_next, wi_next])
                                        i += 1
                                        break
                            else:
                                break  # No improvement found, stop this scan

        return best_tour

    def construct_tour(self, original_tour, delta_path):
        """
        Construct a new tour based on the original tour and the δ-path.

        :param original_tour: List representing the original tour
        :param delta_path: List representing the δ-path
        :return: New tour
        """
        new_tour = original_tour.copy()
        for i in range(0, len(delta_path) - 1, 2):
            u, v = delta_path[i], delta_path[i + 1]
            idx_u = new_tour.index(u)
            idx_v = new_tour.index(v)
            new_tour[idx_u + 1 : idx_v + 1] = reversed(new_tour[idx_u + 1 : idx_v + 1])
        return new_tour

    def is_valid_tour(self, tour):
        """
        Check if the given tour is valid.
        A valid tour must:
        1. Contain all cities exactly once (except the first city, which appears twice - at the start and end)
        2. Start and end with the same city
        3. Only contain valid city indices
        """
        if len(tour) != self.num_cities + 1:
            return False, "Tour length is incorrect"

        if tour[0] != tour[-1]:
            return False, "Tour does not start and end with the same city"

        city_set = set(tour)  # Exclude the last city as it's a repeat of the first

        if len(city_set) != self.num_cities:
            return False, "Tour does not visit all cities exactly once"

        if any(city < 0 or city >= self.num_cities for city in tour):
            return False, "Tour contains invalid city indices"

        # Check if all edges in the tour exist in the graph
        for i in range(len(tour) - 1):
            if (tour[i], tour[i + 1]) not in self.weights:
                return False, f"Invalid edge in tour: {tour[i]} to {tour[i+1]}"

        return True, "Valid Tour"

    def three_opt(self, tour):
        tour = tour[:-1]
        new_tour, new_cost = tour[:], -1

        def better_tour(tour, new_tour):
            new_cost, cost = self.calculate_tour_cost(
                new_tour
            ), self.calculate_tour_cost(tour)
            # print(new_tour, new_cost,"<deciding>",tour, cost)
            return new_tour[:] if cost > new_cost else tour[:]

        for i in range(2, len(tour)):
            for j in range(1, i):
                for k in range(j):
                    new_tour = better_tour(self.swap(tour, i, j, k), new_tour)
                    tour = new_tour[:]
        tour = better_tour(new_tour, tour)
        return tour + [tour[0]]

    def swap(self, tour, i, j, k):
        def better_tour(tour, new_tour):
            new_cost, cost = self.calculate_tour_cost(
                new_tour
            ), self.calculate_tour_cost(tour)
            # print(new_tour, new_cost,":",tour, cost)
            return new_tour[:] if cost > new_cost else tour[:]

        final_tour = tour[:]

        def inner_function(tour, a, b, c, x, y, z):
            new_tour = tour[:]
            new_tour[a], new_tour[b], new_tour[c] = (
                new_tour[x],
                new_tour[y],
                new_tour[z],
            )
            # print(tour, new_tour)
            return better_tour(tour, new_tour)

        A = itertools.permutations((i, j, k), 3)
        perms = list(itertools.permutations(A, 2))

        for (a, b, c), (x, y, z) in perms:
            final_tour = better_tour(final_tour, inner_function(tour, a, b, c, x, y, z))
        # print("returning", final_tour, self.calculate_tour_cost(final_tour))
        return final_tour

    def generate_initial_population(self, population_size):
        """
        Generate an initial population using Christofides and Lin-Kernighan.
        """
        population = []
        for _ in range(population_size):
            tour = self.christofides()
            tour = self.lin_kernighan(tour)
            population.append(tour)
        return population

    def crossover(self, parent1, parent2):
        """
        Perform ordered crossover (OX) between two parent tours.
        """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = parent1[start:end]
        remaining = [item for item in parent2 if item not in child]
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining.pop(0)
        return child

    def mutate(self, tour):
        """
        Perform a simple swap mutation on the tour.
        """
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def genetic_algorithm(self, population_size=50, generations=100, mutation_rate=0.1):
        """
        Implement the genetic algorithm for TSP.
        """
        population = self.generate_initial_population(population_size)

        for _ in range(generations):
            # Evaluate fitness
            fitness = [1 / self.calculate_tour_cost(tour) for tour in population]

            # Select parents
            parents = random.choices(population, weights=fitness, k=population_size)

            # Create new population
            new_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                # Mutation
                if random.random() < mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < mutation_rate:
                    child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Apply Lin-Kernighan to the best individual
            best_tour = min(new_population, key=self.calculate_tour_cost)
            best_tour = self.lin_kernighan(best_tour)

            # Replace the worst individual with the improved best tour
            worst_idx = max(
                range(len(new_population)),
                key=lambda i: self.calculate_tour_cost(new_population[i]),
            )
            new_population[worst_idx] = best_tour

            population = new_population

        return min(population, key=self.calculate_tour_cost)

    def two_opt(self, tour):
        """
        Perform 2-opt optimization on the given tour.
        """
        improved = True
        best_distance = self.calculate_tour_cost(tour)
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue
                    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                    new_distance = self.calculate_tour_cost(new_tour)
                    if new_distance < best_distance:
                        tour = new_tour
                        best_distance = new_distance
                        improved = True
            if improved:
                break
        return tour

    def remove_crossovers(self, tour):
        """
        Remove edge intersections (crossovers) from the tour.
        """

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        improved = True
        while improved:
            improved = False
            for i, j in itertools.combinations(range(len(tour)), 2):
                i_next = (i + 1) % len(tour)
                j_next = (j + 1) % len(tour)
                if i_next == j or j_next == i:
                    continue
                A = self.coordinates[tour[i]]
                B = self.coordinates[tour[i_next]]
                C = self.coordinates[tour[j]]
                D = self.coordinates[tour[j_next]]
                if intersect(A, B, C, D):
                    tour[i_next : j + 1] = reversed(tour[i_next : j + 1])
                    improved = True
                    break
        return tour

    def or_opt(self, tour):
        """
        Perform Or-opt optimization on the given tour.
        """
        from time import time

        start_time = time()

        def better_tour(tour, new_tour):
            new_cost, cost = self.calculate_tour_cost(
                new_tour
            ), self.calculate_tour_cost(tour)
            # print(new_tour, new_cost,"<deciding>",tour, cost)
            return new_tour[:] if cost > new_cost else tour[:], cost > new_cost

        final_tour = tour[:]
        improvement = True
        while improvement:
            # print(self.calculate_tour_cost(final_tour))
            improvement = False
            for i in range(2, len(tour)):
                for j in range(1, i):
                    for k in range(j):
                        final_tour, potential_improvement = better_tour(
                            final_tour, self.shift_segment(final_tour, i, j, k)
                        )
                        improvement = potential_improvement or improvement
                        if time() - start_time > 200:
                            return final_tour
            break
        return final_tour

    def shift_segment(self, tour, i: int, j: int, k: int) -> List[int]:
        # Ensure i, j, k are within bounds
        tour = tour[:-1]
        n = len(tour)
        i, j, k = i % n, j % n, k % n

        # Determine the segment to be shifted
        if i < j:
            segment = tour[i + 1 : j + 1]
        else:
            segment = tour[i + 1 :] + tour[: j + 1]

        # Remove the segment from the tour
        if i < j:
            new_tour = tour[: i + 1] + tour[j + 1 :]
        else:
            new_tour = tour[j + 1 : i + 1]

        # Insert the segment after position k
        insert_pos = (k + 1) % len(new_tour)
        new_tour = new_tour[:insert_pos] + segment + new_tour[insert_pos:]

        return new_tour + [new_tour[0]]

    def simulated_annealing(
        self,
        initial_tour: List[int],
        initial_temperature: float = 100,
        cooling_rate: float = 0.995,
        iterations: int = 10000,
    ) -> List[int]:
        """
        Optimize tour using Simulated Annealing.

        :param initial_tour: Initial tour to optimize
        :param initial_temperature: Starting temperature
        :param cooling_rate: Rate at which temperature decreases
        :param iterations: Number of iterations
        :return: Optimized tour
        """
        current_tour = initial_tour
        current_cost = self.calculate_tour_cost(current_tour)
        best_tour = current_tour
        best_cost = current_cost
        temperature = initial_temperature

        for _ in range(iterations):
            i, j = sorted(random.sample(range(1, len(current_tour) - 1), 2))
            new_tour = (
                current_tour[:i] + current_tour[i : j + 1][::-1] + current_tour[j + 1 :]
            )
            new_cost = self.calculate_tour_cost(new_tour)

            if new_cost < current_cost or random.random() < math.exp(
                (current_cost - new_cost) / temperature
            ):
                current_tour = new_tour
                current_cost = new_cost
                if current_cost < best_cost:
                    best_tour = current_tour
                    best_cost = current_cost

            temperature *= cooling_rate

        return best_tour

    def edge_recombination_crossover(self, parent1, parent2):
        """
        Perform Edge Recombination Crossover (ERX) on two parent tours.

        :param parent1: First parent tour
        :param parent2: Second parent tour
        :return: Offspring tour
        """
        # Create edge table
        edge_table = {i: set() for i in range(self.num_cities)}
        for tour in [parent1, parent2]:
            for i in range(len(tour)):
                current = tour[i]
                next = tour[(i + 1) % len(tour)]
                edge_table[current].add(next)
                edge_table[next].add(current)

        # Generate offspring
        offspring = [random.choice(parent1)]  # Start with a random city
        while len(offspring) < self.num_cities:
            current = offspring[-1]
            if edge_table[current]:
                # Choose the neighbor with the fewest remaining neighbors
                next = min(edge_table[current], key=lambda x: len(edge_table[x]))
            else:
                # If no neighbors, choose a random unvisited city
                unvisited = set(range(self.num_cities)) - set(offspring)
                next = random.choice(list(unvisited))

            offspring.append(next)

            # Remove the current city from all neighbors' lists
            for neighbor in edge_table[current]:
                edge_table[neighbor].discard(current)
            edge_table[current].clear()

        return offspring

    def genetic_algorithm_tsp(self, population_size=100, generations=1000):
        """
        Solve TSP using a genetic algorithm with ERX.

        :param population_size: Size of the population
        :param generations: Number of generations to run
        :return: Best tour found
        """
        # Initialize population with random tours
        population = [
            random.sample(range(self.num_cities), self.num_cities)
            for _ in range(population_size)
        ]

        for _ in range(generations):
            # Evaluate fitness
            fitness = [1 / self.calculate_tour_cost(tour) for tour in population]

            # Select parents
            parents = random.choices(population, weights=fitness, k=population_size)

            # Create new population through crossover
            new_population = []
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    offspring1 = self.edge_recombination_crossover(
                        parents[i], parents[i + 1]
                    )
                    offspring2 = self.edge_recombination_crossover(
                        parents[i + 1], parents[i]
                    )
                    new_population.extend([offspring1, offspring2])
                else:
                    new_population.append(parents[i])

            # Mutate (simple swap mutation)
            for tour in new_population:
                if random.random() < 0.1:  # 10% mutation rate
                    i, j = random.sample(range(self.num_cities), 2)
                    tour[i], tour[j] = tour[j], tour[i]

            population = new_population

        # Return the best tour found
        return min(population, key=self.calculate_tour_cost)

    def check_and_fix_edge_crossovers(self, tour):
        """
        Check for edge crossovers in a TSP tour and fix them.

        :param tour: List representing the tour
        :return: Fixed tour and number of crossovers fixed
        """

        def ccw(A, B, C):
            return (self.coordinates[C][1] - self.coordinates[A][1]) * (
                self.coordinates[B][0] - self.coordinates[A][0]
            ) > (self.coordinates[B][1] - self.coordinates[A][1]) * (
                self.coordinates[C][0] - self.coordinates[A][0]
            )

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        fixed = 0
        n = len(tour)
        for i in range(n):
            for j in range(i + 2, n - 1):
                if intersect(tour[i], tour[(i + 1) % n], tour[j], tour[(j + 1) % n]):
                    # Swap to fix crossover
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    fixed += 1

        return tour


def read_input() -> Graph:
    """
    Read input from standard input and create a Graph object.

    :return: Initialized Graph object
    """
    graph_type = GraphType(input().strip())
    num_cities = int(input().strip())
    graph = Graph(graph_type, num_cities)

    for i in range(num_cities):
        x, y = map(float, input().strip().split())
        graph.add_coordinate(i, (x, y))

    for u in range(num_cities):
        distances = list(map(float, input().strip().split()))
        graph.add_weight(u, distances)

    return graph


def solve_tsp(graph: Graph) -> None:
    """
    Solve the TSP problem using both Christofides algorithm and optimal solver.

    :param graph: Graph object representing the TSP instance
    """
    from time import time

    start = time()
    tour = graph.christofides()
    functions = [
        graph.two_opt,
        graph.or_opt,
        graph.two_opt,
        graph.lin_kernighan,
        graph.two_opt,
        graph.or_opt,
        graph.two_opt,
        graph.lin_kernighan,
        graph.three_opt,
        graph.two_opt,
        graph.get_optimal_tour,
    ]

    print(" ".join(map(str, tour[:-1])))
    for func in functions:
        tour = func(tour)
        print(" ".join(map(str, tour[:-1])))
        print("valid:", graph.is_valid_tour(tour))
        print(graph.calculate_tour_cost(tour))
        print("time:", time() - start)
        print("")
    """
    #print("Christofides Algorithm:")
    christofides_tour = graph.christofides()
    christofides_cost = graph.calculate_tour_cost(christofides_tour)
    print(' '.join(map(str, christofides_tour)))
    #print(f"Cost: {christofides_cost}")
    
    print("\nweird shit Optimized")
    two_opt_tour = graph.two_opt(graph.or_opt(graph.two_opt(graph.lin_kernighan(graph.two_opt(christofides_tour)))))
    two_opt_cost = graph.calculate_tour_cost(two_opt_tour)
    print(f"Tour: {' '.join(map(str, two_opt_tour))}")
    print(f"Valid: {graph.is_valid_tour(two_opt_tour)}")
    print(f"Cost: {two_opt_cost}")
    print("Time:",time() - start)

    if graph.num_cities <= 10:
        print("\nOptimal Tour:")
        optimal_tour, optimal_cost = graph.get_optimal_tour()
        print(f"Tour: {' '.join(map(str, optimal_tour))}")
        print(f"Cost: {optimal_cost}")
    """


def main() -> None:
    """
    Main function to read input, solve TSP, and output results.
    """
    # print("Arnav Rustagi U20220021\nMukundan Gurumurthy U20220056")
    graph = read_input()
    solve_tsp(graph)


if __name__ == "__main__":
    main()
