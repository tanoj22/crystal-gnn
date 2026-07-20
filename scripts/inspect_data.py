import torch

graphs = torch.load("data/dataset.pt", weights_only=False)

print(f"Loaded {len(graphs)} graphs")
print(f"Type of first element: {type(graphs[0])}")

for i, graph in enumerate(graphs[:3]):
    print(f"\nGraph {i}:")
    print(f"  num_nodes: {graph.num_nodes}")
    print(f"  z: {graph.z}")
    print(f"  edge_index shape: {graph.edge_index.shape}")
    print(f"  edge_attr shape: {graph.edge_attr.shape}")
    print(f"  y: {graph.y}")

num_nodes = [g.num_nodes for g in graphs]
ys = [g.y.item() for g in graphs]
zero_edge_count = sum(1 for g in graphs if g.edge_index.shape[1] == 0)

print("\nSanity stats:")
print(f"  min nodes: {min(num_nodes)}, max nodes: {max(num_nodes)}")
print(f"  min y: {min(ys)}, max y: {max(ys)}")
print(f"  graphs with zero edges: {zero_edge_count}")
