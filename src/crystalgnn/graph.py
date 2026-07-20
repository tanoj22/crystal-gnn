import torch
from torch_geometric.data import Data


def structure_to_graph(structure, cutoff=6.0, target=None):
    """Convert a pymatgen Structure into a PyG graph with atomic numbers and neighbor edges."""
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    sources, targets, distances = [], [], []
    for i, neighbors in enumerate(structure.get_all_neighbors(cutoff)):
        for neighbor in neighbors:
            sources.append(i)
            targets.append(neighbor.index)
            distances.append(neighbor.nn_distance)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)

    data = Data(
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(structure),
    )

    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)

    return data
