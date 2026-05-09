"""
1D-CNN sequence classifier for human vs agent trajectories.

Instead of hand-crafted features, this feeds raw trajectory sequences
(x, y, speed, acceleration, curvature) into a small 1D-CNN.

Run: python analysis/sequence_model.py

Requires: pip install torch
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from features import load_trajectories, events_to_arrays

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEQ_LEN = 200          # fixed sequence length (pad or truncate)
N_CHANNELS = 5         # x_norm, y_norm, speed, acceleration, curvature
BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-3
N_FOLDS = 5
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def trajectory_to_sequence(traj):
    """
    Convert a trajectory JSON to a (SEQ_LEN, N_CHANNELS) numpy array.
    Channels: x_norm, y_norm, speed, acceleration, curvature.
    Returns None if insufficient data.
    """
    result = events_to_arrays(traj.get("events", []))
    if result is None:
        return None

    x, y, t = result
    if len(x) < 10:
        return None

    # Normalize x, y to [0, 1] based on canvas size
    canvas = traj.get("canvas_size", {"width": 600, "height": 350})
    x_norm = x / canvas["width"]
    y_norm = y / canvas["height"]

    # Speed
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-3, dt)
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) / dt
    speed = np.concatenate([[0], speed])  # pad to same length

    # Acceleration
    accel = np.diff(speed) / np.where(dt == 0, 1e-3, dt)
    accel = np.concatenate([[0], accel])

    # Curvature
    gx = np.gradient(x)
    gy = np.gradient(y)
    ggx = np.gradient(gx)
    ggy = np.gradient(gy)
    denom = (gx**2 + gy**2)**1.5
    denom = np.where(denom == 0, 1e-9, denom)
    kappa = np.abs(gx * ggy - gy * ggx) / denom

    # Stack channels
    seq = np.stack([x_norm, y_norm, speed, accel, kappa], axis=1)  # (N, 5)

    # Normalize speed, accel, curvature per-trajectory (zero mean, unit var)
    for ch in range(2, N_CHANNELS):
        col = seq[:, ch]
        mu, sigma = col.mean(), col.std()
        if sigma > 1e-9:
            seq[:, ch] = (col - mu) / sigma

    # Pad or truncate to SEQ_LEN
    if len(seq) >= SEQ_LEN:
        seq = seq[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - len(seq), N_CHANNELS))
        seq = np.concatenate([seq, pad], axis=0)

    return seq.astype(np.float32)


def load_all_sequences():
    """Load all trajectories, convert to sequences, return X, y, metadata."""
    X_list = []
    y_list = []
    meta = []

    for source_dir in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, source_dir)
        if not os.path.isdir(full) or source_dir == "logs":
            continue

        is_agent = 0 if source_dir == "human" else 1
        trajs = load_trajectories(full)

        for traj in trajs:
            if not traj.get("completed"):
                continue
            seq = trajectory_to_sequence(traj)
            if seq is None:
                continue
            X_list.append(seq)
            y_list.append(is_agent)
            meta.append({
                "session_id": traj.get("session_id"),
                "source": traj.get("source"),
                "tunnel_id": traj.get("tunnel_id"),
            })

    X = np.array(X_list)  # (N, SEQ_LEN, N_CHANNELS)
    y = np.array(y_list)
    return X, y, meta


class TrajectoryDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (SEQ_LEN, N_CHANNELS)
        label = self.y[idx]

        if self.augment:
            # Random crop: take a random contiguous window and re-pad
            actual_len = (x.abs().sum(dim=1) > 0).sum().item()
            if actual_len > 30:
                crop_len = max(30, int(actual_len * (0.7 + 0.3 * torch.rand(1).item())))
                start = torch.randint(0, max(1, actual_len - crop_len), (1,)).item()
                cropped = x[start:start + crop_len]
                # Re-pad
                if len(cropped) < SEQ_LEN:
                    pad = torch.zeros(SEQ_LEN - len(cropped), N_CHANNELS)
                    cropped = torch.cat([cropped, pad], dim=0)
                x = cropped[:SEQ_LEN]

            # Small Gaussian noise on speed/accel/curvature channels
            noise = torch.randn_like(x[:, 2:]) * 0.05
            x = x.clone()
            x[:, 2:] += noise

        # Transpose to (channels, seq_len) for Conv1d
        return x.T, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrajectoryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pool
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        h = self.conv(x)         # (batch, 64, 1)
        h = h.squeeze(-1)        # (batch, 64)
        return self.fc(h).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_fold(X_train, y_train, X_val, y_val, fold_idx):
    """Train model on one fold, return val predictions and probabilities."""
    train_ds = TrajectoryDataset(X_train, y_train, augment=True)
    val_ds = TrajectoryDataset(X_val, y_val, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TrajectoryCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Class weighting for imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                               dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best and get final predictions
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    print(f"  Fold {fold_idx + 1}: best val AUC = {best_auc:.3f}")
    return np.array(all_probs), np.array(all_labels)


def run_cv(X, y):
    """Run stratified k-fold cross-validation."""
    n_splits = min(N_FOLDS, min(np.bincount(y)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_probs = np.zeros(len(y))
    all_true = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        probs, labels = train_one_fold(
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            fold,
        )
        all_probs[val_idx] = probs
        all_true[val_idx] = labels

    return all_probs, all_true


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Sequence length: {SEQ_LEN}, Channels: {N_CHANNELS}")

    print("\nLoading sequences...")
    X, y, meta = load_all_sequences()
    print(f"  Total: {len(X)} (Human: {(y==0).sum()}, Agent: {(y==1).sum()})")

    if len(np.unique(y)) < 2:
        print("ERROR: Need both human and agent data.")
        sys.exit(1)

    if min(np.bincount(y)) < 3:
        print("WARNING: Very few samples in one class. Results may be unreliable.")

    print(f"\nTraining 1D-CNN with {min(N_FOLDS, min(np.bincount(y)))}-fold CV...")
    probs, true = run_cv(X, y)

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds)
    try:
        auc = roc_auc_score(true, probs)
    except ValueError:
        auc = 0.5

    print("\n" + "=" * 50)
    print("1D-CNN RESULTS")
    print("=" * 50)
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC:  {auc:.3f}")
    print()
    print(classification_report(true, preds, target_names=["human", "agent"]))

    # Save results
    results = {
        "model": "1D-CNN",
        "seq_len": SEQ_LEN,
        "n_channels": N_CHANNELS,
        "n_samples": len(X),
        "n_human": int((y == 0).sum()),
        "n_agent": int((y == 1).sum()),
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
    }
    out_path = os.path.join(OUT_DIR, "sequence_model_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
