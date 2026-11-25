"""Unified model factory and checkpoint compatibility helpers."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

from backbones.resunet import ResUNet


BACKBONES = ("resunet18", "resunet34", "resunet50")


def auto_image_size(backbone: str) -> int:
    return 512


def build_model(config: Mapping[str, Any]):
    name = str(config["backbone"]).lower()
    use_crsm = bool(config.get("use_crsm", False))
    if name not in BACKBONES:
        raise ValueError(f"Unsupported backbone {name}; choose from {BACKBONES}")
    crsm = dict(
        bottleneck_dim=int(config.get("aux_hidden_dim", 256)),
        perturb_T=int(config.get("t_pert", 3)),
        perturb_sigma=float(config.get("sigma", 0.05)),
        perturb_samples=int(config.get("num_perturb", 4)),
        energy_T=int(config.get("t_stab", 2)),
        energy_gamma=float(config.get("gamma", 0.1)),
        alpha=float(config.get("alpha", 0.3)),
        depth=3,
    )
    common = dict(num_classes=2, use_crsm=use_crsm, pretrained_path=config.get("pretrained_path"), **crsm)
    return ResUNet(name=name, pretrained=bool(config.get("pretrained", True)), **common)


def infer_legacy_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(path)
    name = path.name.lower()
    backbone_match = re.search(r"resnet(18|34|50)", name)
    backbone = f"resunet{backbone_match.group(1)}" if backbone_match else "resunet34"
    lambda_match = re.search(r"aux([0-9.]+)", name)
    alpha_match = re.search(r"alpha([0-9.]+)", name)
    config: Dict[str, Any] = {
        "backbone": backbone,
        "use_crsm": "aux" in name,
        "image_size": 512,
        "lambda_aux": float(lambda_match.group(1).rstrip(".")) if lambda_match else 0.01,
        "alpha": float(alpha_match.group(1).rstrip(".")) if alpha_match else 0.5,
        "t_stab": 2,
        "t_pert": 3,
        "sigma": 0.05,
        "num_perturb": 4,
        "gamma": 0.1,
        "pretrained_path": None,
    }
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    warnings.warn(
        f"Legacy checkpoint {path.name} has no saved config. Recovered filename fields and used old defaults for the rest: {config}",
        RuntimeWarning,
    )
    return config


def unpack_checkpoint(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[str, Any]]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw and "config" in raw:
        config = dict(raw["config"])
        if overrides:
            config.update({key: value for key, value in overrides.items() if value is not None})
        return config, raw["model"], raw
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    return infer_legacy_config(path, overrides), state, {"legacy": True}


def load_model_checkpoint(path: str | Path, device: str, overrides: Mapping[str, Any] | None = None, strict: bool = True):
    config, state, metadata = unpack_checkpoint(path, overrides)
    config["pretrained_path"] = None
    # The checkpoint already contains the initialized/trained encoder. Do not
    # download ImageNet weights again during evaluation.
    config["pretrained"] = False
    model = build_model(config)
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=strict)
    return model.to(device), config, metadata
