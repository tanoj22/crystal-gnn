import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch

from crystalgnn.graph import structure_to_graph
from crystalgnn.model import CrystalGNN


def test_crystal_gnn_forward_on_batch():
    structure = Structure(
        Lattice.cubic(4.1),
        ["Cs", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    graphs = [structure_to_graph(structure), structure_to_graph(structure)]
    batch = Batch.from_data_list(graphs)

    model = CrystalGNN()
    model.eval()
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2,)
    assert torch.isfinite(out).all()
