import time
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from typing import List, Tuple
from enum import Enum
import random
import math
import itertools

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
            if w == v: continue
            self.weights[(u, v)] = w

    def kruskal_mst(self):
        mst = []
        parents, rank = list(range(self.num_cities)), [0]*self.num_cities

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
            closest = min(unmatched, key=lambda u: self.weights.get((v, u), float('inf')))
            matching.append([v, closest, self.weights.get((v, closest))])
            unmatched.remove(closest)
        
        return matching

    def get_optimal_tour(self):
        """
        Find the optimal tour using the Held-Karp algorithm (dynamic programming).
        
        :return: Tuple containing the optimal tour and its cost
        """
        n = self.num_cities
        
        # Initialize the DP table
        dp = {}
        
        # Base case: start from city 0 to each other city
        for i in range(1, n):
            dp[(1 << i, i)] = (self.weights.get((0, i), float('inf')), [0, i])
        
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
                        res.append((dp[(prev, j)][0] + self.weights.get((j, last), float('inf')), j))
                    dp[(bits, last)] = min(res)
        
        # Find the optimal tour
        bits = (2**n - 1) - 1  # All bits set except the 0th city
        res = []
        for i in range(1, n):
            res.append((dp[(bits, i)][0] + self.weights.get((i, 0), float('inf')), i))
        opt, parent = min(res)
        
        # Reconstruct the path
        path = [0]
        for i in range(n - 1):
            path.append(parent)
            new_bits = bits & ~(1 << parent)
            _, parent = dp[(bits, parent)]
            bits = new_bits
        path.append(0)
        
        return path, opt


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
            cost += self.weights.get((tour[i], tour[i+1]), float('inf'))
        return cost

    def distance(self, city1, city2):
        if city1 == city2: return 0
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

    def or_opt(self, tour):
        N = len(tour)
        locally_optimal = False

        while not locally_optimal:
            locally_optimal = True

            for segment_len in range(3, 0, -1):
                for pos in range(N):
                    i = pos
                    X1 = tour[i]
                    X2 = tour[(i + 1) % N]
                    
                    j = (i + segment_len) % N
                    Y1 = tour[j]
                    Y2 = tour[(j + 1) % N]

                    for shift in range(segment_len + 1, N):
                        k = (i + shift) % N
                        Z1 = tour[k]
                        Z2 = tour[(k + 1) % N]

                        if self.seg_shift_gain(X1, X2, Y1, Y2, Z1, Z2) > 0:
                            logger.info(f"Tour before shift: {tour}")
                            logger.info(f"i, j, k: {i}, {j}, {k}")
                            logger.info(f"seg len: {segment_len}")
                            self.shift_seg(tour, i, j, k)
                            logger.info(f"Tour after shift: {tour}")
                            locally_optimal = False
                            break  # Exit the innermost loop
                    
                    if not locally_optimal:
                        break  # Exit the middle loop
                
                if not locally_optimal:
                    break  # Exit the outer loop

        return tour

    def shift_seg(self, tour, i, j, k):
        N = len(tour)
        segment_size = (j - i + N) % N
        logger.info(f"segment_size: {segment_size}")
        shift_size = ((k - i + N) - segment_size + N) % N
        offset = (i + 1 + shift_size)

        # Make a copy of the segment before shift
        segment = [tour[(i + 1 + counter) % N] for counter in range(segment_size)]

        # Shift to the left by segment_size all cities between old position
        # of right end of the segment and new position of its left end
        pos = (i + 1) % N
        for _ in range(shift_size):
            tour[pos] = tour[(pos + segment_size) % N]
            pos = (pos + 1) % N

        # Put the copy of the segment into its new place in the tour
        for pos in range(segment_size):
            tour[(offset + pos) % N] = segment[pos]

    def seg_shift_gain(self, x1, x2, y1, y2, z1, z2):
        current = self.distance(x1, x2) + self.distance(y1, y2) + self.distance(z1, z2)
        shifted = self.distance(x1, y2) + self.distance(y1, z2) + self.distance(z1, x2)

        return current - shifted


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
    print("Christofides Algorithm:")
    christofides_tour = graph.christofides()
    christofides_cost = graph.calculate_tour_cost(christofides_tour)
    print(f"Tour: {' '.join(map(str, christofides_tour))}")
    print(f"Cost: {christofides_cost}")
    
    print("\n2-opt Optimized Christofides")
    two_opt_tour = graph.two_opt(deepcopy(christofides_tour))
    two_opt_cost = graph.calculate_tour_cost(two_opt_tour)
    print(f"Tour: {' '.join(map(str, two_opt_tour))}")
    print(f"Cost: {two_opt_cost}")
    
    print("\nOr-opt Optimized Christofides")
    or_opt_tour = graph.or_opt(deepcopy(two_opt_tour))
    or_opt_cost = graph.calculate_tour_cost(or_opt_tour)
    print(f"Tour: {' '.join(map(str, or_opt_tour))}")
    print(f"Cost: {or_opt_cost}")
    
    # print("\n3-opt Optimized Christofides")
    # three_opt_tour = graph.three_opt(deepcopy(christofides_tour))
    # three_opt_cost = graph.calculate_tour_cost(three_opt_tour)
    # print(f"Tour: {' '.join(map(str, three_opt_tour))}")
    # print(f"Cost: {three_opt_cost}")

    if graph.num_cities <= 10:
        print("\nOptimal Tour:")
        optimal_tour, optimal_cost = graph.get_optimal_tour()
        print(f"Tour: {' '.join(map(str, optimal_tour))}")
        print(f"Cost: {optimal_cost}")

def main() -> None:
    """
    Main function to read input, solve TSP, and output results.
    """
    print("Arnav Rustagi U20220021\nMukundan Gurumurthy U20220056")
    graph = read_input()
    solve_tsp(graph)

if __name__ == "__main__":
    main()
