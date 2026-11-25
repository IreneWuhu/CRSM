"""Evaluate one checkpoint on all four held-out OD/OC domains."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_optic import make_loader, target_domains
from metrics import case_metrics, summarize_cases
from models import BACKBONES, load_model_checkpoint


@torch.no_grad()
def evaluate_targets(
    model,
    source_domain: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    device: str,
    metadata: Mapping[str, object] | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    model.eval()
    domain_rows: List[Dict[str, object]] = []
    all_cases: List[Dict[str, object]] = []
    metadata = dict(metadata or {})
    for domain in target_domains(source_domain):
        loader = make_loader(data_root, domain, image_size, batch_size, workers, seed, train=False)
        domain_cases: List[Dict[str, object]] = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            output = model(image, compute_aux=False)
            logits = output["logits"]
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            rows = case_metrics(logits, target, batch["case_id"])
            for row in rows:
                row.update(metadata)
                row["target_domain"] = domain
            domain_cases.extend(rows)
        summary = summarize_cases(domain_cases)
        domain_rows.append({"target_domain": domain, **summary})
        all_cases.extend(domain_cases)
    average = {key: float(np.mean([float(row[key]) for row in domain_rows])) for key in ("disc_dice", "cup_dice", "mean_dice", "disc_asd", "cup_asd")}
    domain_rows.append({"target_domain": "TARGET_MEAN", **average})
    return domain_rows, all_cases


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--source_domain", choices=list("ABCDE"), default=None)
    parser.add_argument("--backbone", choices=BACKBONES, default=None, help="legacy-checkpoint override")
    parser.add_argument("--image_size", type=int, default=None, help="legacy-checkpoint override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"backbone": args.backbone, "image_size": args.image_size}
    model, config, checkpoint = load_model_checkpoint(args.checkpoint, args.device, overrides, strict=True)
    source = args.source_domain or config.get("source_domain")
    if source not in list("ABCDE"):
        raise ValueError("source_domain is required for legacy checkpoints")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    metadata = {
        "run_id": config.get("run_id", Path(args.checkpoint).stem),
        "method": "crsm" if config.get("use_crsm") else "baseline",
        "backbone": config["backbone"],
        "source_domain": source,
        "seed": config.get("seed", -1),
        "best_epoch": checkpoint.get("epoch", -1),
    }
    domains, cases = evaluate_targets(
        model, source, args.data_root, int(config["image_size"]), args.batch_size,
        args.num_workers, int(config.get("seed", 0)), args.device, metadata,
    )
    write_csv(output_dir / "test_case_metrics_best.csv", cases)
    write_csv(output_dir / "test_domain_metrics_best.csv", domains)
    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "config": config, "domains": domains}
    (output_dir / "test_best_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in domains:
        print(row)


if __name__ == "__main__":
    main()
