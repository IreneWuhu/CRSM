"""Training-only CRSM auxiliary dynamics and losses."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AuxiliaryCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 3):
        super().__init__()
        if depth != 3:
            raise ValueError("The released CRSM uses a three-layer auxiliary CNN")
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
    def __init__(self, input_dim=512, bottleneck_dim=256, perturb_T=3,
                 perturb_sigma=0.05, perturb_samples=4, energy_T=2,
                 energy_gamma=0.1, alpha=0.3, depth=3):
        super().__init__()
        self.dynamics_model = AuxiliaryCNN(input_dim, bottleneck_dim, depth)
        self.bn = nn.BatchNorm2d(input_dim)
        self.perturb_T = perturb_T
        self.perturb_sigma = perturb_sigma
        self.perturb_samples = perturb_samples
        self.energy_T = energy_T
        self.energy_gamma = energy_gamma
        self.alpha = alpha

    def _step(self, state: Tensor):
        update = self.dynamics_model(state)
        if update.shape[1] == 1 and state.shape[1] != 1:
            update = update.expand(-1, state.shape[1], -1, -1)
        return state + update, update

    def perturbation_loss(self, x: Tensor) -> Tensor:
        losses = []
        for _ in range(self.perturb_samples):
            perturbed = x + torch.randn_like(x) * self.perturb_sigma
            for _ in range(self.perturb_T):
                perturbed, _ = self._step(perturbed)
            losses.append((perturbed - x).square().mean())
        return torch.stack(losses).mean()

    def stability_loss(self, x: Tensor) -> Tensor:
        state = x
        energy_penalties = []
        for _ in range(self.energy_T):
            next_state, _ = self._step(state)
            energy_delta = next_state.square().mean() - state.square().mean()
            energy_penalties.append(F.relu(energy_delta + self.energy_gamma))
            state = next_state
        return torch.stack(energy_penalties).mean()

    def forward(self, feature: Tensor):
        x = self.bn(feature)
        loss_pert = self.perturbation_loss(x)
        loss_stab = self.stability_loss(x)
        return {
            "loss_aux": self.alpha * loss_pert + (1.0 - self.alpha) * loss_stab,
            "loss_pert": loss_pert,
            "loss_stab": loss_stab,
        }
