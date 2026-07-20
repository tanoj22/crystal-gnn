from dotenv import load_dotenv
import os
from pathlib import Path

import torch
from mp_api.client import MPRester

from crystalgnn.graph import structure_to_graph

load_dotenv()

api_key = os.getenv("MP_API_KEY")
if not api_key:
    raise ValueError("MP_API_KEY not found in environment. Set it in your .env file.")

with MPRester(api_key) as mpr:
    docs = mpr.materials.summary.search(
        fields=["material_id", "structure", "formation_energy_per_atom"],
        num_chunks=100,
        chunk_size=1000,
    )

graphs = []
skipped = 0
for doc in docs:
    if doc.structure is not None and doc.formation_energy_per_atom is not None:
        graph = structure_to_graph(doc.structure, target=doc.formation_energy_per_atom)
        if graph.edge_index.shape[1] == 0:
            skipped += 1
            continue
        graphs.append(graph)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
torch.save(graphs, data_dir / "dataset.pt")

print(f"Saved {len(graphs)} graphs to data/dataset.pt")
print(f"Skipped {skipped} graphs with zero edges")
