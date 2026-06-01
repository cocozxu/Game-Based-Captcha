"""
PPO fine-tuning for the trace-the-tunnel RL agent.

Starts from the BC checkpoint and fine-tunes with:
  - PPO clipped objective (ε=0.2)
  - GAE advantage estimation (γ=0.99, λ=0.95)
  - Entropy bonus for exploration
  - Anti-detection regularization: penalises deviation from human
    feature distributions (centerline_dev, speed profile, path efficiency)

Run:
  cd trace-the-tunnel
  source .venv/bin/activate
  python experiments_rl/ppo_train.py [--iters 200] [--rollout 2048]

Checkpoints are saved to experiments_rl/models/ppo_policy_<iter>.pt
and the best (highest episode return) to ppo_policy_best.pt.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from policy import TunnelPolicy, save_policy, load_policy
from tunnel_env import TunnelEnv, OBS_DIM, CANVAS_W, CANVAS_H, TUNNEL_WIDTH

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "tunnel_config.json")

# ---------------------------------------------------------------------------
# PPO hyper-parameters
# ---------------------------------------------------------------------------
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.05
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
LR = 2e-4
EPOCHS_PER_ITER = 8
MINIBATCH = 256


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, rollout_len: int):
        self.rollout_len = rollout_len
        self.clear()

    def clear(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_advantages(self, last_value: float):
        """GAE advantage estimation."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values = self.values + [last_value]

        for t in reversed(range(n)):
            delta = self.rewards[t] + GAMMA * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + GAMMA * LAM * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def as_tensors(self, returns, advantages):
        obs_t = torch.from_numpy(np.array(self.obs, dtype=np.float32))
        act_t = torch.from_numpy(np.array(self.actions, dtype=np.float32))
        lp_t = torch.tensor(self.log_probs, dtype=torch.float32)
        ret_t = torch.from_numpy(returns)
        adv_t = torch.from_numpy(advantages)
        return obs_t, act_t, lp_t, ret_t, adv_t


# ---------------------------------------------------------------------------
# Anti-detection regularization
# ---------------------------------------------------------------------------

def human_feature_stats():
    """Target distributions computed from features.csv, is_agent==0 rows (n=187)."""
    return {
        # (mean, std) derived from human-only traces
        "cl_dev_abs": (11.24, 1.89),     # pixels off centerline
        "speed_px_ms": (0.50, 0.07),     # pixels per ms
        "efficiency": (0.78, 0.08),      # path_length / straight_line_dist
    }


def compute_trajectory_features(xs, ys, ts, centerline):
    """Extract key features from a generated trajectory for anti-detection reward."""
    if len(xs) < 5:
        return None

    from tunnel_env import dist_to_centerline
    cl_devs = [dist_to_centerline(x, y, centerline) for x, y in zip(xs, ys)]
    cl_mean = float(np.mean(cl_devs))

    dxs = np.diff(xs)
    dys = np.diff(ys)
    dts = np.diff(ts)
    dts = np.where(dts == 0, 1e-3, dts)
    speeds = np.sqrt(dxs**2 + dys**2) / dts
    speed_mean = float(np.mean(speeds))

    path_len = float(np.sum(np.sqrt(dxs**2 + dys**2)))
    sld = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    efficiency = sld / path_len if path_len > 0 else 0.0

    return {"cl_dev_abs": cl_mean, "speed_px_ms": speed_mean, "efficiency": efficiency}


def anti_detection_penalty(feats: dict, human_stats: dict) -> float:
    """
    Returns a penalty (negative reward) for deviating from human feature distributions.
    Dead zone within 1-sigma so early non-human-like successes still get positive reward.
    """
    if feats is None:
        return 0.0
    penalty = 0.0
    for key, (h_mean, h_std) in human_stats.items():
        if key not in feats:
            continue
        z = abs((feats[key] - h_mean) / (h_std + 1e-6))
        if z > 1.0:
            penalty += (z - 1.0) ** 2 * 0.2
    return -penalty


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def collect_episode(env: TunnelEnv, policy: TunnelPolicy, deterministic: bool = False):
    """Run one episode, returning (obs_list, act_list, log_prob_list, reward_list, value_list, done_list, info)."""
    obs, _ = env.reset()
    obs_list, act_list, lp_list, rew_list, val_list, done_list = [], [], [], [], [], []
    total_reward = 0.0

    # Trajectory for anti-detection reward (computed once at episode end)
    xs, ys = [env._x], [env._y]
    ts = [0.0]  # simulated timestamps in ms

    done = truncated = False

    while not (done or truncated):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = policy.get_action(obs_t, deterministic=deterministic)
        action_np = action.squeeze(0).numpy()
        log_prob_np = float(log_prob.item())
        value_np = float(value.item())

        obs_new, reward, done, truncated, info = env.step(action_np)

        # Track trajectory for post-episode anti-detection reward
        xs.append(env._x)
        ys.append(env._y)
        ts.append(ts[-1] + 8.3)  # simulate 8.3ms per step

        obs_list.append(obs)
        act_list.append(action_np)
        lp_list.append(log_prob_np)
        rew_list.append(reward)
        val_list.append(value_np)
        done_list.append(done or truncated)

        total_reward += reward
        obs = obs_new

    # Anti-detection reward: distribute evenly across all steps for better credit assignment
    if done and info.get("reason") == "success" and len(xs) > 5:
        feats = compute_trajectory_features(
            np.array(xs), np.array(ys), np.array(ts), env.centerline
        )
        ad_bonus = anti_detection_penalty(feats, human_feature_stats())
        ad_per_step = ad_bonus / len(rew_list)
        rew_list = [r + ad_per_step for r in rew_list]
        total_reward += ad_bonus

    return obs_list, act_list, lp_list, rew_list, val_list, done_list, info, total_reward


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(policy: TunnelPolicy, optimizer, obs_t, act_t, lp_old_t, ret_t, adv_t, ent_coef: float):
    """One PPO update over the full rollout buffer."""
    n = obs_t.shape[0]
    indices = torch.randperm(n)
    val_old_t = ret_t.detach()  # used for value clipping

    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    n_batches = 0

    for _ in range(EPOCHS_PER_ITER):
        for start in range(0, n, MINIBATCH):
            idx = indices[start:start + MINIBATCH]
            obs_b = obs_t[idx]
            act_b = act_t[idx]
            lp_old_b = lp_old_t[idx]
            ret_b = ret_t[idx]
            adv_b = adv_t[idx]

            # Normalize advantages per minibatch
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            log_prob_new, entropy, value_new = policy.evaluate_actions(obs_b, act_b)

            # Skip NaN batches (numerical instability guard)
            if torch.isnan(log_prob_new).any() or torch.isnan(value_new).any():
                continue

            # Policy loss (clipped PPO objective)
            ratio = torch.exp(torch.clamp(log_prob_new - lp_old_b, -5.0, 5.0))
            pg1 = ratio * adv_b
            pg2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
            pg_loss = -torch.min(pg1, pg2).mean()

            # Value loss (clipped)
            vf_loss = nn.functional.mse_loss(value_new, ret_b)

            # Entropy bonus
            ent_loss = -entropy.mean()

            loss = pg_loss + VF_COEF * vf_loss + ent_coef * ent_loss

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.mean().item()
            n_batches += 1

        # Re-shuffle each epoch
        indices = torch.randperm(n)

    if n_batches == 0:
        return {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}

    return {
        "pg_loss": total_pg_loss / n_batches,
        "vf_loss": total_vf_loss / n_batches,
        "entropy": total_entropy / n_batches,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_ppo(
    n_iters: int = 200,
    rollout_len: int = 2048,
    bc_checkpoint: str = "bc_policy_best.pt",
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    all_tunnel_ids = [t["tunnel_id"] for t in config["tunnels"]]

    # Load BC checkpoint or initialize fresh
    bc_path = os.path.join(MODEL_DIR, bc_checkpoint)
    if os.path.exists(bc_path):
        print(f"Loading BC checkpoint: {bc_path}")
        policy = load_policy(bc_path)
        policy.train()
    else:
        print(f"BC checkpoint not found at {bc_path}. Initializing fresh policy.")
        print("  Run bc_train.py first for best results.")
        policy = TunnelPolicy()

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    buffer = RolloutBuffer(rollout_len)
    best_mean_return = -float("inf")

    print(f"\nPPO training: {n_iters} iterations, rollout={rollout_len}")
    print(f"  Tunnels: {all_tunnel_ids}")
    print()

    # Rotating environment — cycle through all tunnels for diversity
    env_pool = [TunnelEnv(tid, seed=i) for i, tid in enumerate(all_tunnel_ids)]
    env_idx = 0

    for iteration in range(1, n_iters + 1):
        t_start = time.time()
        buffer.clear()
        ep_returns = []
        ep_successes = 0
        ep_count = 0

        # Collect rollout_len steps
        steps_collected = 0
        while steps_collected < rollout_len:
            env = env_pool[env_idx % len(env_pool)]
            env_idx += 1

            obs_list, act_list, lp_list, rew_list, val_list, done_list, info, ep_ret = \
                collect_episode(env, policy)

            for t in range(len(obs_list)):
                buffer.add(
                    obs_list[t], act_list[t], lp_list[t],
                    rew_list[t], val_list[t], done_list[t]
                )

            steps_collected += len(obs_list)
            ep_returns.append(ep_ret)
            ep_count += 1
            if info.get("reason") == "success":
                ep_successes += 1

        # Bootstrap last value for GAE
        last_obs = torch.from_numpy(env_pool[0]._compute_obs()).unsqueeze(0)
        with torch.no_grad():
            _, last_val = policy(last_obs)
        last_value = float(last_val.item())

        returns, advantages = buffer.compute_returns_advantages(last_value)
        obs_t, act_t, lp_t, ret_t, adv_t = buffer.as_tensors(returns, advantages)

        # PPO update — entropy decays linearly from ENTROPY_COEF to 0.005
        ent_coef = max(0.005, ENTROPY_COEF * (1 - iteration / n_iters))
        policy.train()
        losses = ppo_update(policy, optimizer, obs_t, act_t, lp_t, ret_t, adv_t, ent_coef)
        policy.eval()

        mean_ret = float(np.mean(ep_returns))
        success_rate = ep_successes / max(ep_count, 1)
        elapsed = time.time() - t_start

        print(
            f"iter {iteration:4d}/{n_iters}  "
            f"ep={ep_count:3d}  sr={100*success_rate:5.1f}%  "
            f"ret={mean_ret:6.2f}  "
            f"pg={losses['pg_loss']:.4f}  vf={losses['vf_loss']:.4f}  "
            f"ent={losses['entropy']:.3f}  "
            f"t={elapsed:.1f}s"
        )

        if mean_ret > best_mean_return:
            best_mean_return = mean_ret
            save_policy(policy, os.path.join(MODEL_DIR, "ppo_policy_best.pt"))

        if iteration % 25 == 0:
            save_policy(policy, os.path.join(MODEL_DIR, f"ppo_policy_{iteration:04d}.pt"))

    save_policy(policy, os.path.join(MODEL_DIR, "ppo_policy_final.pt"))
    print(f"\nPPO done. Best mean return={best_mean_return:.2f}")
    print(f"  Best checkpoint: models/ppo_policy_best.pt")
    return policy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of PPO iterations")
    parser.add_argument("--rollout", type=int, default=2048,
                        help="Steps per rollout buffer")
    parser.add_argument("--bc_checkpoint", type=str, default="bc_policy_best.pt",
                        help="BC checkpoint filename (inside models/)")
    args = parser.parse_args()

    train_ppo(
        n_iters=args.iters,
        rollout_len=args.rollout,
        bc_checkpoint=args.bc_checkpoint,
    )
