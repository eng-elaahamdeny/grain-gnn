import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import json

NUM_GRAINS = 200
GRID_SIZE  = 100
np.random.seed(42)

points = np.random.uniform(0, GRID_SIZE, size=(NUM_GRAINS, 2))

vor = Voronoi(points)

grains = []

for i, point in enumerate(points):
    region_index = vor.point_region[i]
    vertex_indices = vor.regions[region_index]

    if -1 in vertex_indices or len(vertex_indices) == 0:
        continue

    vertices = vor.vertices[vertex_indices]

    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    diameter = 2 * np.sqrt(area / np.pi)

    neighbors = []
    for ridge_points in vor.ridge_points:
        if i in ridge_points:
            neighbor = ridge_points[0] if ridge_points[1] == i else ridge_points[1]
            neighbors.append(int(neighbor))

    orientation = np.random.uniform(0, 180)

    grains.append({
        "id":            i,
        "x":             float(point[0]),
        "y":             float(point[1]),
        "area":          float(area),
        "diameter":      float(diameter),
        "num_neighbors": len(neighbors),
        "orientation":   float(orientation),
        "neighbors":     neighbors
    })

areas       = [g["area"] for g in grains]
median_area = np.median(areas)

for g in grains:
    g["label"] = 1 if g["area"] > median_area else 0

with open("grains.json", "w") as f:
    json.dump(grains, f, indent=2)

print(f"Generated {len(grains)} grains")
print(f"Median area: {median_area:.2f}")
print(f"Grains that will grow:   {sum(g['label']==1 for g in grains)}")
print(f"Grains that will shrink: {sum(g['label']==0 for g in grains)}")

fig, ax = plt.subplots(figsize=(8, 8))

for g in grains:
    color = "#e74c3c" if g["label"] == 1 else "#3498db"
    ax.plot(g["x"], g["y"], "o", color=color, markersize=4)

for simplex in vor.ridge_vertices:
    if -1 in simplex:
        continue
    p1, p2 = vor.vertices[simplex[0]], vor.vertices[simplex[1]]
    if (0 <= p1[0] <= GRID_SIZE and 0 <= p1[1] <= GRID_SIZE and
        0 <= p2[0] <= GRID_SIZE and 0 <= p2[1] <= GRID_SIZE):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", linewidth=0.5)

ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_title("Synthetic Grain Microstructure\nRed = will grow | Blue = will shrink")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("microstructure.png", dpi=150)
plt.show()
print("Saved microstructure.png")