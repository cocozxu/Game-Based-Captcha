"""
Actor-Critic policy for the trace-the-tunnel RL agent.

Simple Gaussian policy — no tanh squash. Actions are clamped to
[-MAX_STEP_PX, MAX_STEP_PX] in the environment. Log-probabilities are
computed under the unbounded Gaussian (standard PPO practice for
bounded action spaces without tanh, avoids atanh instability).
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from tunnel_env import OBS_DIM, MAX_STEP_PX

ACT_DIM = 2
HIDDEN = 128
LOG_STD_INIT = -1.0    # initial std ≈ e^{-1} ≈ 0.37 × MAX_STEP_PX ≈ 5px
LOG_STD_MIN = -3.0
LOG_STD_MAX = 0.5


class TunnelPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden: int = HIDDEN,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor: predicts mean in pixel space
        self.actor_mean = nn.Linear(hidden, act_dim)
        # State-independent log-std
        self.log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))

        # Critic: predicts state value
        self.critic = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _dist(self, obs: torch.Tensor) -> Normal:
        h = self.backbone(obs)
        mean = self.actor_mean(h)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def forward(self, obs: torch.Tensor):
        """Returns (action_mean, value)."""
        h = self.backbone(obs)
        mean = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        return mean, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample an action and return (action, log_prob, value).

        action  : (B, 2) in pixels; no clamping here — env.step() handles it
        log_prob: (B,)
        value   : (B,)
        """
        dist = self._dist(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)

        h = self.backbone(obs)
        value = self.critic(h).squeeze(-1)

        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate stored actions under the current policy.

        actions : (B, 2) in pixels (as sampled, before env clamping)
        Returns (log_prob, entropy, value).
        """
        dist = self._dist(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        h = self.backbone(obs)
        value = self.critic(h).squeeze(-1)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_policy(policy: TunnelPolicy, path: str):
    torch.save(policy.state_dict(), path)


def load_policy(path: str, device: str = "cpu") -> TunnelPolicy:
    policy = TunnelPolicy()
    policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    policy.eval()
    return policy
