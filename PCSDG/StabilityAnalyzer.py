"""Training-only CRSM loss module for PCSDG experiments."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AuxiliaryNet(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StabilityAnalyzer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, t_pert=3, t_stab=2,
                 sigma=0.05, num_perturb=4, gamma=0.1, alpha=0.3):
        super().__init__()
        self.dynamics_model = AuxiliaryNet(input_dim, hidden_dim)
        self.bn = nn.BatchNorm2d(input_dim)
        self.t_pert = t_pert
        self.t_stab = t_stab
        self.sigma = sigma
        self.num_perturb = num_perturb
        self.gamma = gamma
        self.alpha = alpha

    def step(self, state: Tensor):
        update = self.dynamics_model(state)
        update = update.expand(-1, state.shape[1], -1, -1)
        return state + update, update

    def perturbation_loss(self, x: Tensor) -> Tensor:
        losses = []
        for _ in range(self.num_perturb):
            state = x + torch.randn_like(x) * self.sigma
            for _ in range(self.t_pert):
                state, _ = self.step(state)
            losses.append((state - x).square().mean())
        return torch.stack(losses).mean()

    def stability_loss(self, x: Tensor) -> Tensor:
        state = x
        penalties = []
        for _ in range(self.t_stab):
            next_state, _ = self.step(state)
            energy_delta = next_state.square().mean() - state.square().mean()
            penalties.append(F.relu(energy_delta + self.gamma))
            state = next_state
        return torch.stack(penalties).mean()

    def forward(self, feature: Tensor):
        normalized = self.bn(feature)
        loss_pert = self.perturbation_loss(normalized)
        loss_stab = self.stability_loss(normalized)
        return {
            "loss_aux": self.alpha * loss_pert + (1.0 - self.alpha) * loss_stab,
            "loss_pert": loss_pert,
            "loss_stab": loss_stab,
        }
