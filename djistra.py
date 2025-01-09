import heapq

class Solution:
    def shortestPath(self, n, edges, src):
        # Create an adjacency list from the edges
        graph = {i: [] for i in range(n)}
        for u, v, w in edges:
            graph[u].append((v, w))
        
        # Initialize distances and priority queue
        distances = {i: float('inf') for i in range(n)}
        distances[src] = 0
        priority_queue = [(0, src)]  # (distance, vertex)
        
        # Dijkstra's algorithm
        while priority_queue:
            curr_dist, curr_vertex = heapq.heappop(priority_queue)
            
            # Skip if the current distance is not the shortest
            if curr_dist > distances[curr_vertex]:
                continue
            
            # Relax neighboring vertices
            for neighbor, weight in graph[curr_vertex]:
                distance = curr_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # Convert unreachable vertices to -1
        for vertex in range(n):
            if distances[vertex] == float('inf'):
                distances[vertex] = -1
        
        return {k: distances[k] for k in range(n)}

# Example usage
n = 5
edges = [[0, 1, 10], [0, 2, 3], [1, 3, 2], [2, 1, 4], [2, 3, 8], [2, 4, 2], [3, 4, 5]]
src = 0

solution = Solution()
shortest_paths = solution.shortestPath(n, edges, src)
print(shortest_paths)  # Expected output: {0: 0, 1: 7, 2: 3, 3: 9, 4: 5}
