import random
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from crystalgnn.model import CrystalGNN

graphs = torch.load("data/dataset.pt", weights_only=False)

random.seed(42)
random.shuffle(graphs)

n = len(graphs)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train_set = graphs[:n_train]
val_set = graphs[n_train : n_train + n_val]
test_set = graphs[n_train + n_val :]

device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = device == "cuda"

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=pin_memory)
val_loader = DataLoader(val_set, batch_size=64, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=64, pin_memory=pin_memory)

model = CrystalGNN().to(device)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)
loss_fn = torch.nn.L1Loss()

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
checkpoint_path = models_dir / "cgnn.pt"


def evaluate(loader):
    model.eval()
    total_error = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            total_error += (pred - batch.y.view(-1)).abs().sum().item()
            total_graphs += batch.num_graphs
    return total_error / total_graphs


best_val_mae = float("inf")
best_epoch = 0
epochs_since_improvement = 0
patience = 30
min_delta = 1e-4

for epoch in range(1, 501):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    train_mae = total_loss / total_graphs
    val_mae = evaluate(val_loader)
    scheduler.step(val_mae)
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"epoch {epoch}  train_MAE {train_mae:.4f}  "
        f"val_MAE {val_mae:.4f}  lr {current_lr:.6g}"
    )

    if val_mae < best_val_mae - min_delta:
        best_val_mae = val_mae
        best_epoch = epoch
        epochs_since_improvement = 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}: best val_MAE {best_val_mae:.4f} "
                f"at epoch {best_epoch}"
            )
            break
        if current_lr < 1e-6:
            print(
                f"Stopping at epoch {epoch}: learning rate {current_lr:.6g} < 1e-6; "
                f"best val_MAE {best_val_mae:.4f} at epoch {best_epoch}"
            )
            break

model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
test_mae = evaluate(test_loader)
print(f"Best val_MAE {best_val_mae:.4f} at epoch {best_epoch}")
print(f"test_MAE {test_mae:.4f}")
