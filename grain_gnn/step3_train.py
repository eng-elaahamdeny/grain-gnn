import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt

# ── LOAD GRAPH ─────────────────────────────────────────────────────────────────
data = torch.load("graph.pt", weights_only=False)
print(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")

# make edges undirected (both directions)
data.edge_index = to_undirected(data.edge_index)

# ── TRAIN/TEST SPLIT ───────────────────────────────────────────────────────────
num_nodes = data.num_nodes
indices   = torch.randperm(num_nodes)
train_end = int(0.8 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[indices[:train_end]] = True
test_mask[indices[train_end:]]  = True

print(f"Training grains: {train_mask.sum().item()}")
print(f"Testing grains:  {test_mask.sum().item()}")

# ── GNN MODEL ──────────────────────────────────────────────────────────────────
class GrainGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # layer 1: aggregate neighbor info
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # layer 2: refine
        x = self.conv2(x, edge_index)
        return x

model     = GrainGNN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"\nModel architecture:")
print(model)

# ── TRAINING LOOP ──────────────────────────────────────────────────────────────
train_losses = []
test_accuracies = []

for epoch in range(200):
    # --- train ---
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # --- evaluate ---
    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

    train_losses.append(loss.item())
    test_accuracies.append(test_acc)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test Accuracy: {test_acc*100:.1f}%")

print(f"\nFinal Test Accuracy: {test_acc*100:.1f}%")

# ── PLOT RESULTS ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, color="#e74c3c")
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2.plot(test_accuracies, color="#3498db")
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig("training.png", dpi=150)
plt.show()
print("Saved training.png")