import random
import heapq
import tkinter as tk
from tabulate import tabulate


class Maze:
    def __init__(self):
        self.size = 6
        self.nodes = [[Node(x, y) for x in range(self.size)] for y in range(self.size)]
        self.start = None
        self.goal = None
        self.barriers = []

    def set_start(self, x, y):
        self.start = self.nodes[y][x]

    def set_goal(self, x, y):
        self.goal = self.nodes[y][x]

    def add_barrier(self, x, y):
        self.barriers.append(self.nodes[y][x])

    def visualize_maze(self, path=None, show_costs=False, title="Maze Visualization"):
        window = tk.Tk()
        window.title(title)
        cell_size = 80  # Size of each cell in the grid
        canvas = tk.Canvas(window, width=self.size * cell_size, height=self.size * cell_size)
        canvas.pack()

        # Define colors for different cell types
        colors = {
            ' ': 'white',  # Open passage
            'B': 'black',  # Barrier
            'S': 'green',  # Starting node
            'G': 'blue',  # Goal node
            '*': 'yellow'  # Path
        }

        # Draw the maze
        for row in self.nodes:
            for node in row:
                x, y = node.x, node.y
                cell_type = ' '
                if node in self.barriers:
                    cell_type = 'B'
                elif node == self.start:
                    cell_type = 'S'
                elif node == self.goal:
                    cell_type = 'G'
                if path and node in path and node not in [self.start, self.goal]:
                    cell_type = '*'

                color = colors.get(cell_type, 'white')
                canvas.create_rectangle(x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size,
                                        fill=color)
                if show_costs:
                    cost = manhattan_distance(node, self.goal)
                    canvas.create_text(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2, text=str(cost))

        window.mainloop()


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = []

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Node({self.x}, {self.y})"


def initialize_maze(maze):
    # Randomly select starting node from the first two rows (0-11)
    start_row = random.randint(0, 5)
    start_col = random.randint(0, 1)
    start_node = start_row * maze.size + start_col

    # Randomly select goal node from the last two rows (24-35)
    goal_row = random.randint(0, 5)
    goal_col = random.randint(4, 5)
    goal_node = goal_row * maze.size + goal_col

    maze.set_start(start_col, start_row)
    maze.set_goal(goal_col, goal_row)

    # Randomly select four barrier nodes, excluding the start and goal nodes
    available_indices = list(set(range(36)) - {start_node, goal_node})
    barrier_indices = random.sample(available_indices, 4)
    for index in barrier_indices:
        maze.add_barrier(index % 6, index // 6)

    # Define neighbors for each node
    for y in range(maze.size):
        for x in range(maze.size):
            node = maze.nodes[y][x]
            # Add neighbors (horizontal, vertical, diagonal)
            temp_neighbors = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if 0 <= x + dx < maze.size and 0 <= y + dy < maze.size and (dx != 0 or dy != 0):
                        neighbor = maze.nodes[y + dy][x + dx]
                        if neighbor not in maze.barriers:
                            temp_neighbors.append(neighbor)
            # Sort neighbors in increasing order based on their (x, y) values
            node.neighbors = sorted(temp_neighbors, key=lambda n: (n.x, n.y))


maze = Maze()
initialize_maze(maze)


def dfs(maze, current_node, visited, goal):
    if current_node in visited:
        return False
    visited.append(current_node)
    if current_node == goal:
        return True
    for neighbor in current_node.neighbors:
        if dfs(maze, neighbor, visited, goal):
            return True
    return False


def run_dfs(maze):
    visited = []
    if dfs(maze, maze.start, visited, maze.goal):
        time_taken = len(visited)  # Each node takes 1 minute
        return visited, time_taken
    else:
        return None, 0


def manhattan_distance(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)


def a_star_search(maze):
    start = maze.start
    goal = maze.goal
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for row in maze.nodes for node in row}
    g_score[start] = 0
    f_score = {node: float('inf') for row in maze.nodes for node in row}
    f_score[start] = manhattan_distance(start, goal)

    visited = set()  # Set to track visited nodes

    while open_set:
        current = heapq.heappop(open_set)[1]
        visited.add(current)  # Add current node to visited set

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            time_taken = len(visited)  # Each node takes 1 minute
            return path, visited, time_taken

        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal)
                if neighbor not in visited:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, set(), 0

# Function to calculate the mean and variance of time taken and path length
def calculate_mean_variance(data):
    mean = sum(data) / len(data)
    variance = sum([(x - mean) ** 2 for x in data]) / len(data)
    return mean, variance

# Running the algorithms multiple times and collecting data
num_iterations = 100
dfs_times = []
dfs_path_lengths = []
a_star_times = []
a_star_path_lengths = []

for _ in range(num_iterations):
    maze = Maze()
    initialize_maze(maze)

    # DFS
    dfs_path, dfs_time = run_dfs(maze)
    if dfs_path:
        dfs_times.append(dfs_time)
        dfs_path_lengths.append(len(dfs_path))

    # A*
    a_star_path, _, a_star_time = a_star_search(maze)
    if a_star_path:
        a_star_times.append(a_star_time)
        a_star_path_lengths.append(len(a_star_path))


def print_maze_with_tabulate(maze, path=None):
    maze_map = [[' ' for _ in range(maze.size)] for _ in range(maze.size)]

    for barrier in maze.barriers:
        maze_map[barrier.y][barrier.x] = '\033[91mB\033[0m'

    maze_map[maze.start.y][maze.start.x] = '\033[93mS\033[0m'
    maze_map[maze.goal.y][maze.goal.x] = '\033[93mG\033[0m'

    if path:
        for node in path:
            if node != maze.start and node != maze.goal:
                maze_map[node.y][node.x] = '\033[92m*\033[0m'

    # Convert the maze map to a format suitable for tabulate
    table = [[f"{cell}" for cell in row] for row in maze_map]
    print(tabulate(table, tablefmt="mixed_grid"))


dfs_visited, dfs_time_taken = run_dfs(maze)
a_star_path, a_star_visited, a_star_time_taken = a_star_search(maze)

print("DFS Visited Nodes:", dfs_visited)
print("DFS Time Taken :", dfs_time_taken, "minutes")
print("\nDFS Path in Maze:")
print_maze_with_tabulate(maze, dfs_visited)

print("A* Visited Nodes:", a_star_visited)
print("\nA* Visited Cells in Maze:")
print_maze_with_tabulate(maze, a_star_visited)

print("\nA* Final Path:", a_star_path)
print("A* Time Taken :", a_star_time_taken, "minutes")
print("\nA* Final Path in Maze:")
print_maze_with_tabulate(maze, a_star_path)

# Visualize the same maze with DFS and A* paths
maze.visualize_maze(dfs_visited, title="DFS Path")
maze.visualize_maze(a_star_path, show_costs=True, title="A* Path")

# Calculate mean and variance for DFS
dfs_time_mean, dfs_time_variance = calculate_mean_variance(dfs_times)
dfs_path_length_mean, dfs_path_length_variance = calculate_mean_variance(dfs_path_lengths)

# Calculate mean and variance for A*
a_star_time_mean, a_star_time_variance = calculate_mean_variance(a_star_times)
a_star_path_length_mean, a_star_path_length_variance = calculate_mean_variance(a_star_path_lengths)

print("************************************************")
print("DFS Algorithm")
print("Mean solution time = ", dfs_time_mean)
print("Variance of solution time = ", dfs_time_variance)
print("Mean path length = ", dfs_path_length_mean )
print("Variance of path length = " , dfs_path_length_variance)

print("================================================")
print("A* Algorithm")
print("Mean solution time = ", a_star_time_mean)
print("Variance of solution time = ",  a_star_time_variance)
print("Mean path length = ", a_star_path_length_mean )
print("Variance of path length = " , a_star_path_length_variance)