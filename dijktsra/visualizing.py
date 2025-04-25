import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import heapq
import matplotlib.image as img
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

graph = {
    1: {2: 6, 3: 2, 5: 1},
    2: {1: 6, 3: 5, 4: 1},
    3: {1: 2, 2: 3, 4: 1},
    4: {2: 1, 3: 4, 5: 2},
    5: {1: 1, 4: 2}
}

G = nx.Graph()
for node, neighbors in graph.items():
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

def dijkstra(graph, start):
    distances = {start: 0}
    visited = set()
    prev = {start: None}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node not in visited:
            visited.add(current_node)
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                prev[neighbor] = current_node

    return distances, prev

def get_path(prev, start, end):
    path = []
    while end is not None:
        path.append(end)
        end = prev[end]
    path.reverse()
    return path

shortest_paths, prev = dijkstra(graph, 1)
print("Shortest paths:", shortest_paths)

all_nodes = list(graph.keys())
combined_path = []
node = 1
visited_nodes = set()

while len(visited_nodes) < len(all_nodes):
    shortest_paths, prev = dijkstra(graph, node)
    next_node = min((x for x in all_nodes if x not in visited_nodes), key=lambda x: shortest_paths[x])
    path_segment = get_path(prev, node, next_node)
    combined_path.extend(path_segment[1:] if combined_path else path_segment)
    visited_nodes.update(path_segment)
    node = next_node

print("Combined Path:", combined_path)

fig, ax = plt.subplots(figsize=(7, 7))

pos = {
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
    4: (0, 0),
    5: (0.5, 0.5)
}

labels = nx.get_edge_attributes(G, 'weight')

def draw_curved_edges(ax, G, pos):
    for u, v, d in G.edges(data=True):
        u_pos, v_pos = np.array(pos[u]), np.array(pos[v])
        control_point = (u_pos + v_pos) / 2 + np.array([0.1, 0.1])
        path_data = [
            (Path.MOVETO, pos[u]),
            (Path.CURVE3, control_point),
            (Path.CURVE3, pos[v])
        ]
        codes, verts = zip(*path_data)
        path = Path(verts, codes)
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor='black', linewidth=2))

def draw_graph(ax, G, pos):
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1400, ax=ax)
    draw_curved_edges(ax, G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

def mark_path(ax, path, pos):
    for i in range(len(path) - 1):
        u_pos, v_pos = np.array(pos[path[i]]), np.array(pos[path[i+1]])
        control_point = (u_pos + v_pos) / 2 + np.array([0.1, 0.1])
        path_data = [
            (Path.MOVETO, pos[path[i]]),
            (Path.CURVE3, control_point),
            (Path.CURVE3, pos[path[i+1]])
        ]
        codes, verts = zip(*path_data)
        ax.add_patch(PathPatch(Path(verts, codes), facecolor='none', edgecolor='red', linewidth=4))

def update(num):
    ax.clear()
    draw_graph(ax, G, pos)
    mark_path(ax, combined_path[:num+1], pos)

ani = anm.FuncAnimation(fig, update, frames=len(combined_path), interval=1000)
ani.save('dijkstra_animation.gif', writer='pillow')

plt.show()