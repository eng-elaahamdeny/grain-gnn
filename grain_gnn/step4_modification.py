import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt

# ── LOAD GRAPH ─────────────────────────────────────────────────────────────────
data = torch.load("graph.pt", weights_only=False)
data.edge_index = to_undirected(data.edge_index)

# ── TRAIN/TEST SPLIT (same as before) ─────────────────────────────────────────
torch.manual_seed(42)
num_nodes = data.num_nodes
indices   = torch.randperm(num_nodes)
train_end = int(0.8 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[indices[:train_end]] = True
test_mask[indices[train_end:]]  = True

# ── BASELINE GCN (same as step 3) ─────────────────────────────────────────────
from torch_geometric.nn import GCNConv

class GrainGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 64)
        self.conv2 = GCNConv(64, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ── MODIFIED GAT ───────────────────────────────────────────────────────────────
class GrainGAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # heads=4 means 4 attention mechanisms running in parallel
        self.conv1 = GATConv(data.num_node_features, 16, heads=4, dropout=0.3)
        # concat=False means we average the 4 heads at the end
        self.conv2 = GATConv(64, 2, heads=1, concat=False, dropout=0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ── TRAINING FUNCTION ──────────────────────────────────────────────────────────
def train_model(model, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    losses        = []
    accuracies    = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out  = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            acc  = (pred[test_mask] == data.y[test_mask]).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

    return losses, accuracies

# ── TRAIN BOTH ─────────────────────────────────────────────────────────────────
print("Training baseline GCN...")
torch.manual_seed(42)
gcn_model = GrainGCN()
gcn_losses, gcn_accs = train_model(gcn_model)
print(f"GCN  Final Accuracy: {gcn_accs[-1]*100:.1f}%")

print("Training modified GAT...")
torch.manual_seed(42)
gat_model = GrainGAT()
gat_losses, gat_accs = train_model(gat_model)
print(f"GAT  Final Accuracy: {gat_accs[-1]*100:.1f}%")

print(f"\nImprovement: {(gat_accs[-1] - gcn_accs[-1])*100:.1f}%")

# ── PLOT COMPARISON ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(gcn_losses, color="#e74c3c", label="GCN (baseline)")
ax1.plot(gat_losses, color="#2ecc71", label="GAT (modified)")
ax1.set_title("Training Loss: GCN vs GAT")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(gcn_accs, color="#e74c3c", label="GCN (baseline)")
ax2.plot(gat_accs, color="#2ecc71", label="GAT (modified)")
ax2.set_title("Test Accuracy: GCN vs GAT")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("modification_comparison.png", dpi=150)
plt.show()
print("Saved modification_comparison.png")