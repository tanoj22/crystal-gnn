# Crystal GNN

A crystal graph neural network that predicts materials properties directly from crystal structure, trained on Materials Project data. This project predicts formation energy per atom with a test MAE of 0.049 eV/atom, near chemical accuracy. It is a from-scratch reimplementation of the CGCNN approach (Xie & Grossman, 2018).

## Result

![Predicted vs actual formation energy](results/pred_vs_actual.png)

*Test MAE 0.049 eV/atom on a held-out test set of ~10,000 (9,999) crystals from ~99,997 Materials Project structures. Predictions track DFT formation energies across the full energy range.*

## How it works

A crystal is converted to a graph: atoms are nodes (stored as atomic number), nearby atoms within a 6 Angstrom cutoff are connected by edges (including periodic images), and each edge carries the interatomic distance. The model embeds each atom, expands edge distances into Gaussian bins, and runs 3 rounds of message passing (CGConv layers) so each atom encodes its local environment. All atom vectors are pooled into one crystal-level vector, and a small MLP head predicts the target property.

## Project structure

| Path | Description |
|------|-------------|
| `src/crystalgnn/graph.py` | Converts a pymatgen `Structure` into a PyG `Data` graph |
| `src/crystalgnn/model.py` | `CrystalGNN` model (embedding, Gaussian expansion, CGConv, pooling, head) |
| `scripts/fetch_data.py` | Fetches structures and formation energies from Materials Project |
| `scripts/train.py` | Trains the model on cached graphs |
| `scripts/plot_results.py` | Evaluates on the test split and saves a prediction plot |
| `scripts/inspect_data.py` | Prints dataset statistics |
| `tests/` | Unit tests for graph construction and model forward pass |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Create a `.env` file in the project root with your Materials Project API key:

```
MP_API_KEY=your_key_here
```

Get a key at [materialsproject.org](https://materialsproject.org).

## Usage

```bash
python scripts/fetch_data.py      # fetch and cache graphs to data/dataset.pt
python scripts/train.py           # train model, save best checkpoint to models/cgnn.pt
python scripts/plot_results.py    # evaluate test split and save results/pred_vs_actual.png
python -m pytest                  # run tests
```

## Model details

~75,000 trainable parameters. CGCNN-style architecture with 3 message-passing rounds, Gaussian edge expansion, and global mean pooling. Trained to convergence with early stopping, gradient clipping, and a ReduceLROnPlateau learning rate scheduler on ~99,997 Materials Project crystals.

## Notes

Formation energy is used here as a validation task to confirm the pipeline against published CGCNN accuracy. The same graph construction and model can be pointed at other crystal properties by changing the target field in `fetch_data.py` and retraining.
