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

    print("\nOptimal Tour:")
    optimal_tour, optimal_cost = graph.get_optimal_tour()
    print(f"Tour: {' '.join(map(str, optimal_tour))}")
    print(f"Cost: {optimal_cost}")

def main() -> None:
    """
    Main function to read input, solve TSP, and output results.
    """
    graph = read_input()
    solve_tsp(graph)

if __name__ == "__main__":
    main()