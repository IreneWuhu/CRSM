"""Losses and case-level OD/OC metrics."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import Tensor, nn


class SegmentationLoss(nn.Module):
    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        prob = torch.sigmoid(logits)
        intersection = (prob * target).flatten(1).sum(1)
        denominator = prob.flatten(1).sum(1) + target.flatten(1).sum(1)
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return dice_loss + self.bce(logits, target)


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    denom = float(pred.sum() + target.sum())
    return 100.0 if denom == 0.0 else 200.0 * float(np.logical_and(pred, target).sum()) / denom


def _asd(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(np.hypot(*pred.shape))
    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    target_surface = np.logical_xor(target, binary_erosion(target))
    distance_to_target = distance_transform_edt(~target_surface)
    distance_to_pred = distance_transform_edt(~pred_surface)
    distances = np.concatenate([distance_to_target[pred_surface], distance_to_pred[target_surface]])
    return float(distances.mean()) if distances.size else 0.0


def case_metrics(logits: Tensor, target: Tensor, case_ids: Iterable[str], threshold: float = 0.5) -> List[Dict[str, float | str]]:
    pred = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy()
    ref = (target >= 0.5).detach().cpu().numpy()
    rows: List[Dict[str, float | str]] = []
    for index, case_id in enumerate(case_ids):
        disc_dice = _dice(pred[index, 0], ref[index, 0])
        cup_dice = _dice(pred[index, 1], ref[index, 1])
        rows.append({
            "case_id": str(case_id),
            "disc_dice": disc_dice,
            "cup_dice": cup_dice,
            "mean_dice": (disc_dice + cup_dice) / 2.0,
            "disc_asd": _asd(pred[index, 0], ref[index, 0]),
            "cup_asd": _asd(pred[index, 1], ref[index, 1]),
        })
    return rows


def summarize_cases(rows: List[Dict[str, float | str]]) -> Dict[str, float]:
    keys = ("disc_dice", "cup_dice", "mean_dice", "disc_asd", "cup_asd")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
