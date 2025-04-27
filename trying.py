import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import heapq
import numpy as np
from scipy.spatial import distance_matrix
from itertools import permutations
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Rastgele noktalar oluştur
np.random.seed(42)
points = np.random.rand(10, 2)

# Mesafe matrisi oluştur
dist_matrix = distance_matrix(points, points)

# Brute-force TSP çözümü
n = len(points)
min_path = None
min_length = float('inf')

for perm in permutations(range(n)):
    length = sum(dist_matrix[perm[i], perm[i+1]] for i in range(n - 1))
    length += dist_matrix[perm[-1], perm[0]]
    if length < min_length:
        min_length = length
        min_path = perm

# Ölçek: 1 birim = 100 metre
scale = 100
min_length_metre = min_length * scale
path_points = points[list(min_path) + [min_path[0]]]

# TSP noktalarının pozisyonlarını belirle
pos_tsp = {i: (points[i][0], points[i][1]) for i in range(len(points))}
combined_tsp_path = list(min_path) + [min_path[0]]

# Dijkstra Grafiği için rastgele ağırlıklar oluştur
graph = {i: {} for i in range(n)}
for i in range(n):
    for j in range(i + 1, n):
        weight = np.random.randint(1, 10)
        graph[i][j] = weight
        graph[j][i] = weight

# Dijkstra Algoritması
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    prev = {node: None for node in graph}
    priority_queue = [(0, start)]
    visited = set()

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                prev[neighbor] = current_node

    return distances, prev

# En kısa yolları hesapla
shortest_paths, prev = dijkstra(graph, 0)

# Grafiği oluştur
G = nx.Graph()
for node, neighbors in graph.items():
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

# Figure ayarları
fig, ax = plt.subplots(figsize=(8, 8))

# Eğri çizgileri ekleyelim
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

# Grafiği çiz
def draw_graph(ax, G, pos):
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, ax=ax)
    draw_curved_edges(ax, G, pos)

# TSP yolunu işaretle
def mark_tsp_path(ax, path, pos):
    for i in range(len(path) - 1):
        u_pos, v_pos = np.array(pos[path[i]]), np.array(pos[path[i+1]])
        control_point = (u_pos + v_pos) / 2 + np.array([0.1, 0.1])
        path_data = [
            (Path.MOVETO, pos[path[i]]),
            (Path.CURVE3, control_point),
            (Path.CURVE3, pos[path[i+1]])
        ]
        codes, verts = zip(*path_data)
        ax.add_patch(PathPatch(Path(verts, codes), facecolor='none', edgecolor='red', linewidth=3))

# Animasyonu güncelle
def update(num):
    ax.clear()
    draw_graph(ax, G, pos_tsp)
    mark_tsp_path(ax, combined_tsp_path[:num+1], pos_tsp)

# Animasyonu çalıştır
ani = anm.FuncAnimation(fig, update, frames=len(combined_tsp_path), interval=1000)
ani.save('tsp_dijkstra_animation.gif', writer='pillow')

plt.show()

# En kısa yol sırasını yazdır
print("TSP En Kısa Yol Sırası:")
for i in range(len(min_path)):
    print(f"{min_path[i]} -> ", end="")
print(min_path[0])

# Koordinatları yazdır
print("\nKoordinatlar (gidiş sırasına göre):")
for idx in list(min_path) + [min_path[0]]:
    print(f"{idx}: {points[idx]}")