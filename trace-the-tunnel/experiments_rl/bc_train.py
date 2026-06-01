"""
Behavioral Cloning (BC) pretraining.

Loads all completed human trajectories, reconstructs each tunnel in Python,
converts each consecutive event pair into an (observation, action) sample,
and trains the actor head via supervised MSE loss.

This gives the PPO a warm-start in the human motion distribution before
any RL exploration, so it doesn't have to discover reasonable paths from
scratch.

Run:
  cd trace-the-tunnel
  source .venv/bin/activate
  python experiments_rl/bc_train.py [--epochs 100] [--lr 3e-4]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow imports from this directory
sys.path.insert(0, os.path.dirname(__file__))
from policy import TunnelPolicy, save_policy, OBS_DIM, MAX_STEP_PX
from tunnel_env import (
    TunnelEnv, OBS_DIM, generate_tunnel, signed_dist_to_centerline,
    is_inside_tunnel, TUNNEL_WIDTH, CANVAS_W, CANVAS_H,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "human")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "tunnel_config.json")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _compute_obs_from_state(
    x, y, vx, vy, centerline, end_x, end_y
):
    """Recomputes the 10-dim observation from raw trajectory state."""
    cl_signed, arc_prog = signed_dist_to_centerline(x, y, centerline)
    cl_abs = abs(cl_signed)

    goal_dx = end_x - x
    goal_dy = end_y - y
    goal_dist = math.hypot(goal_dx, goal_dy)

    return np.array([
        x / CANVAS_W,
        y / CANVAS_H,
        vx / MAX_STEP_PX,
        vy / MAX_STEP_PX,
        cl_signed / TUNNEL_WIDTH,
        goal_dx / CANVAS_W,
        goal_dy / CANVAS_H,
        goal_dist / 700.0,
        arc_prog,
        (TUNNEL_WIDTH - cl_abs) / TUNNEL_WIDTH,
    ], dtype=np.float32)


def build_dataset(data_dir: str, seed_map: dict, min_points: int = 10):
    """
    Scan human JSONs, return (observations, actions) as numpy arrays.
    action = (dx, dy) in pixels — next step displacement.
    """
    obs_list, act_list = [], []
    skipped, loaded = 0, 0

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            traj = json.load(f)

        if not traj.get("completed"):
            skipped += 1
            continue

        tid = traj.get("tunnel_id")
        if tid not in seed_map:
            skipped += 1
            continue

        moves = [e for e in traj["events"] if e["event_type"] == "mousemove"]
        if len(moves) < min_points:
            skipped += 1
            continue

        # Reconstruct tunnel geometry for this trajectory
        _, _, centerline = generate_tunnel(seed_map[tid])
        end_x = centerline[-1].x
        end_y = centerline[-1].y

        vx, vy = 0.0, 0.0
        for i in range(len(moves) - 1):
            e0, e1 = moves[i], moves[i + 1]
            x0, y0 = float(e0["x"]), float(e0["y"])
            x1, y1 = float(e1["x"]), float(e1["y"])

            # Skip events that were outside the tunnel (shouldn't be many)
            if not is_inside_tunnel(x0, y0, centerline):
                vx, vy = 0.0, 0.0
                continue

            obs = _compute_obs_from_state(x0, y0, vx, vy, centerline, end_x, end_y)
            dx = x1 - x0
            dy = y1 - y0

            obs_list.append(obs)
            act_list.append(np.array([dx, dy], dtype=np.float32))

            vx, vy = dx, dy

        loaded += 1

    print(f"  Loaded {loaded} trajectories, skipped {skipped}")
    print(f"  Total (obs, action) pairs: {len(obs_list)}")

    return np.array(obs_list), np.array(act_list)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bc(epochs: int = 150, lr: float = 3e-4, batch_size: int = 512):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    seed_map = {t["tunnel_id"]: t["seed"] for t in config["tunnels"]}

    print("Building dataset...")
    obs_np, act_np = build_dataset(DATA_DIR, seed_map)
    if len(obs_np) == 0:
        print("ERROR: No training data found. Ensure data/human/ has completed trajectories.")
        sys.exit(1)

    # Clip actions to action space
    act_np = np.clip(act_np, -MAX_STEP_PX, MAX_STEP_PX)

    # Compute dataset statistics for diagnostics
    print(f"  action dx: mean={act_np[:,0].mean():.3f} std={act_np[:,0].std():.3f}")
    print(f"  action dy: mean={act_np[:,1].mean():.3f} std={act_np[:,1].std():.3f}")
    step_sizes = np.linalg.norm(act_np, axis=1)
    print(f"  step size: mean={step_sizes.mean():.3f} std={step_sizes.std():.3f} "
          f"p5={np.percentile(step_sizes,5):.3f} p95={np.percentile(step_sizes,95):.3f}")

    obs_t = torch.from_numpy(obs_np)
    act_t = torch.from_numpy(act_np)
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    policy = TunnelPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    print(f"\nTraining BC for {epochs} epochs, lr={lr}, batch={batch_size} ...")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs_b, act_b in loader:
            pred_mean, _ = policy(obs_b)
            # Clamp target to what the policy can actually produce
            act_b_clamped = torch.clamp(act_b, -MAX_STEP_PX, MAX_STEP_PX)
            loss = nn.functional.mse_loss(pred_mean, act_b_clamped)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_policy(policy, os.path.join(MODEL_DIR, "bc_policy_best.pt"))

    save_policy(policy, os.path.join(MODEL_DIR, "bc_policy_final.pt"))
    print(f"\nBC done. Best loss={best_loss:.5f}")
    print(f"  Saved: models/bc_policy_best.pt  and  models/bc_policy_final.pt")
    return policy


# ---------------------------------------------------------------------------
# Quick evaluation: run the BC policy on one tunnel in simulation
# ---------------------------------------------------------------------------

def quick_eval(policy_path: str, tunnel_id: int = 0, n_trials: int = 10):
    from policy import load_policy
    print(f"\nQuick eval: {n_trials} trials on tunnel {tunnel_id} ...")
    pol = load_policy(policy_path)
    successes = 0
    for i in range(n_trials):
        env = TunnelEnv(tunnel_id=tunnel_id, seed=i)
        obs, _ = env.reset()
        done = trunc = False
        while not (done or trunc):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                action, _, _ = pol.get_action(obs_t, deterministic=True)
            obs, rew, done, trunc, info = env.step(action.squeeze(0).numpy())
        if info.get("reason") == "success":
            successes += 1
    print(f"  Success rate: {successes}/{n_trials} ({100*successes/n_trials:.0f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval", action="store_true", help="Quick eval after training")
    args = parser.parse_args()

    policy = train_bc(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    if args.eval:
        for tid in range(10):
            quick_eval(
                os.path.join(MODEL_DIR, "bc_policy_best.pt"),
                tunnel_id=tid,
                n_trials=5,
            )
