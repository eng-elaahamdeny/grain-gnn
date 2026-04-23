# GNN-Based Grain Growth Prediction

## Problem Statement
In materials science, metals are made of thousands of tiny crystal regions called grains.
When metal is heated, some grains grow and others shrink — a process called grain growth.
Predicting which grains will grow is important for understanding material strength and durability.

This project represents a 2D metal microstructure as a graph and trains a Graph Neural Network
to predict whether each grain will grow or shrink.

## Dataset
A synthetic microstructure was generated using Voronoi tessellation — a standard method
that produces grain shapes statistically similar to real metal microstructures.
200 seed points were scattered randomly on a 100x100 grid. Each seed became the center
of one grain, with Voronoi boundaries forming the grain walls. 184 complete grains
were retained after removing incomplete border grains.

## Graph Construction
- **Nodes:** Each grain is a node (184 nodes total)
- **Edges:** Two grains are connected if they share a boundary (1004 edges total)
- **Node features:** 4 features per grain:
  - Area — size of the grain
  - Equivalent diameter — width if the grain were a circle
  - Number of neighbors — how many grains touch it
  - Orientation — crystal direction angle (0 to 180 degrees)
- **Edge features:** 2 features per boundary:
  - Boundary length — distance between grain centers
  - Misorientation — difference in orientation angle between two grains

## Feature Justification
**Why area and diameter?**
Larger grains have less curved boundaries, which means lower surface energy.
Nature always moves toward lower energy states, so large grains are energetically
favored to grow at the expense of smaller ones. This is called capillarity-driven grain growth.

**Why number of neighbors?**
Grains with more neighbors have more boundaries through which they can exchange material.
A grain with many neighbors has more pathways for growth or shrinkage.

**Why orientation and misorientation?**
Grain boundaries between crystals with very different orientations (high misorientation)
have higher energy and are more mobile — they move faster. This directly affects
which grains grow and which shrink.

## Labels
- Label 1 (grow): grains with area above the median
- Label 0 (shrink): grains with area below the median
- This gives a balanced dataset: 92 grow, 92 shrink

## Model Architecture
A 2-layer Graph Convolutional Network (GCN):
- Layer 1: GCNConv(4 → 64) with ReLU activation and 30% dropout
- Layer 2: GCNConv(64 → 2) outputting grow/shrink scores
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)
- Epochs: 200

**How the GNN uses neighborhood information:**
Each grain aggregates feature information from all its neighbors before making a prediction.
This means a grain's prediction is influenced not just by its own size, but also by
the sizes and orientations of the grains surrounding it — which is physically meaningful.

## Results
- Training grains: 147 (80%)
- Testing grains: 37 (20%)
- **Final Test Accuracy: 75.7%**

## Modification: GCN vs GAT
The baseline GCN treats all neighbors equally when aggregating information.
We replaced it with a Graph Attention Network (GAT) which learns to assign
different importance weights to different neighbors.

**Physical motivation:** In real grain growth, a grain is more strongly influenced
by its largest neighbor than its smallest one. GAT can learn this asymmetry
while GCN cannot.

**Architecture:** GATConv with 4 parallel attention heads in layer 1,
single head in layer 2.

**Result:** Both models achieved comparable accuracy (~70%), but GCN showed
more stable training. GAT showed higher variance, likely due to the small
dataset size — attention mechanisms need more data to learn reliable weights.
This is itself a meaningful finding: for small synthetic datasets, simpler
models like GCN are more appropriate, while GAT would likely outperform
on larger real microstructure datasets.

## Connection to MD Simulation Work
This pipeline directly extends to atomistic simulation data from LAMMPS.
Instead of Voronoi grains, nodes would represent atoms or clusters.
Instead of grain area, node features would include potential energy,
coordination number, and local stress. The same GNN architecture could
then predict material properties like diffusion coefficients or
mechanical strength without running new simulations each time.

## File Structure
- step1_generate.py — generates synthetic microstructure
- step2_build_graph.py — builds the graph from grain data
- step3_train.py — trains the baseline GCN
- step4_modification.py — compares GCN vs GAT
- grains.json — generated grain data
- graph.pt — saved PyTorch Geometric graph
- microstructure.png — visualization of grains
- graph.png — visualization of grain graph
- training.png — loss and accuracy curves
- modification_comparison.png — GCN vs GAT comparison