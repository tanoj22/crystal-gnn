import torch
from pymatgen.core import Lattice, Structure

from crystalgnn.graph import structure_to_graph


def test_structure_to_graph_for_cesium_chloride():
    structure = Structure(
        Lattice.cubic(4.1),
        ["Cs", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    graph = structure_to_graph(structure)

    assert graph.num_nodes == 2
    assert torch.equal(graph.z, torch.tensor([55, 17], dtype=torch.long))
    assert graph.edge_index.shape[1] > 0
    assert torch.all(graph.edge_attr > 0)


def test_structure_to_graph_with_target():
    structure = Structure(
        Lattice.cubic(4.1),
        ["Cs", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    graph = structure_to_graph(structure, target=1.23)

    assert torch.isclose(graph.y, torch.tensor([1.23], dtype=torch.float))
