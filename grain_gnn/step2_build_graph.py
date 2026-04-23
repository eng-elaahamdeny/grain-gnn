import json
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx

with open("grains.json", "r") as f:
    grains = json.load(f)

print(f"Loaded {len(grains)} grains")

id_to_index = {g["id"]: i for i, g in enumerate(grains)}

node_features = []

for g in grains:
    node_features.append([
        g["area"],
        g["diameter"],
        g["num_neighbors"],
        g["orientation"]
    ])

node_features = torch.tensor(node_features, dtype=torch.float)
node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)

print(f"Node feature matrix shape: {node_features.shape}")

edge_src      = []
edge_dst      = []
edge_features = []

for g in grains:
    if g["id"] not in id_to_index:
        continue
    i = id_to_index[g["id"]]

    for neighbor_id in g["neighbors"]:
        if neighbor_id not in id_to_index:
            continue
        j = id_to_index[neighbor_id]

        gi = grains[i]
        gj = grains[j]
        dx = gi["x"] - gj["x"]
        dy = gi["y"] - gj["y"]
        boundary_length = np.sqrt(dx**2 + dy**2)

        misorientation = abs(gi["orientation"] - gj["orientation"])
        if misorientation > 90:
            misorientation = 180 - misorientation

        edge_src.append(i)
        edge_dst.append(j)
        edge_features.append([boundary_length, misorientation])

edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
edge_attr  = torch.tensor(edge_features, dtype=torch.float)
edge_attr  = (edge_attr - edge_attr.mean(dim=0)) / (edge_attr.std(dim=0) + 1e-8)

print(f"Edge index shape:   {edge_index.shape}")
print(f"Edge feature shape: {edge_attr.shape}")

labels = torch.tensor([g["label"] for g in grains], dtype=torch.long)
print(f"Labels shape: {labels.shape}")

data = Data(
    x          = node_features,
    edge_index = edge_index,
    edge_attr  = edge_attr,
    y          = labels
)

print(f"\nGraph summary:")
print(f"  Nodes (grains):     {data.num_nodes}")
print(f"  Edges (boundaries): {data.num_edges}")
print(f"  Node features:      {data.num_node_features}")
print(f"  Edge features:      {data.num_edge_features}")

torch.save(data, "graph.pt")
print("\nSaved graph.pt")

G         = nx.Graph()
positions = {i: (grains[i]["x"], grains[i]["y"]) for i in range(len(grains))}
colors    = ["#e74c3c" if g["label"] == 1 else "#3498db" for g in grains]

for i in range(len(grains)):
    G.add_node(i)
for src, dst in zip(edge_src, edge_dst):
    G.add_edge(src, dst)

plt.figure(figsize=(8, 8))
nx.draw(G,
        pos=positions,
        node_color=colors,
        node_size=30,
        width=0.5,
        edge_color="#cccccc")
plt.title("Grain Graph\nRed = will grow | Blue = will shrink")
plt.savefig("graph.png", dpi=150)
plt.show()
print("Saved graph.png")