"""Train one single-source OD/OC baseline or CRSM run and test every epoch."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_optic import make_loader
from metrics import SegmentationLoss
from models import BACKBONES, auto_image_size, build_model
from test import evaluate_targets, write_csv


SEEDS = (0, 42, 2026)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("crsm_train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def train_dice(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    intersection = (pred * target).flatten(2).sum(2)
    denominator = pred.flatten(2).sum(2) + target.flatten(2).sum(2)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6) * 100.0
    return {"disc": float(dice[:, 0].mean()), "cup": float(dice[:, 1].mean())}


def append_mean(total: Dict[str, float], values: Dict[str, float], weight: int) -> None:
    total["weight"] += weight
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + value * weight


def mean_totals(total: Dict[str, float]) -> Dict[str, float]:
    weight = max(total.pop("weight"), 1.0)
    return {key: value / weight for key, value in total.items()}


def read_history(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if key == "target_domain":
                    parsed[key] = value
                elif key == "epoch":
                    parsed[key] = int(float(value))
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
            rows.append(parsed)
    return rows


def save_curves(path: Path, train_rows: List[Dict[str, object]], test_rows: List[Dict[str, object]], best_epoch: int) -> None:
    epochs = [int(row["epoch"]) for row in train_rows]
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    panels = axes.ravel()
    panels[0].plot(epochs, [row["loss_total"] for row in train_rows]); panels[0].set_title("Total loss")
    panels[1].plot(epochs, [row["loss_seg"] for row in train_rows]); panels[1].set_title("Segmentation loss")
    for key in ("loss_aux", "loss_pert", "loss_stab"):
        panels[2].plot(epochs, [row[key] for row in train_rows], label=key)
    panels[2].set_title("Auxiliary losses"); panels[2].legend()
    for key in ("train_disc_dice", "train_cup_dice", "train_mean_dice"):
        panels[3].plot(epochs, [row[key] for row in train_rows], label=key)
    panels[3].set_title("Training DSC"); panels[3].legend()
    domain_rows = [row for row in test_rows if row["target_domain"] != "TARGET_MEAN"]
    for domain in sorted({str(row["target_domain"]) for row in domain_rows}):
        rows = [row for row in domain_rows if row["target_domain"] == domain]
        panels[4].plot([row["epoch"] for row in rows], [row["mean_dice"] for row in rows], label=domain)
    panels[4].set_title("Target-domain mean DSC"); panels[4].legend()
    average_rows = [row for row in test_rows if row["target_domain"] == "TARGET_MEAN"]
    if average_rows:
        test_epochs = [row["epoch"] for row in average_rows]
        panels[5].plot(test_epochs, [row["mean_dice"] for row in average_rows], label="mean")
        panels[5].axvline(best_epoch, color="red", linestyle="--", label=f"best={best_epoch}")
        panels[5].set_title("Four-target average DSC"); panels[5].legend()
        panels[6].plot(test_epochs, [row["disc_dice"] for row in average_rows], label="disc")
        panels[6].plot(test_epochs, [row["cup_dice"] for row in average_rows], label="cup")
        panels[6].set_title("Test Disc/Cup DSC"); panels[6].legend()
        panels[7].plot(test_epochs, [row["disc_asd"] for row in average_rows], label="disc")
        panels[7].plot(test_epochs, [row["cup_asd"] for row in average_rows], label="cup")
        panels[7].set_title("Test ASD"); panels[7].legend()
    panels[8].plot(epochs, [row["learning_rate"] for row in train_rows])
    panels[8].axvline(best_epoch, color="red", linestyle="--")
    panels[8].set_title("Learning rate")
    for axis in panels:
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.set_xlabel("Epoch")
    fig.suptitle("CRSM OPTIC training and target-oracle evaluation")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_domain", choices=list("ABCDE"), required=True)
    parser.add_argument("--backbone", choices=BACKBONES, default="resunet34")
    parser.add_argument("--use_crsm", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=0, help="0 selects the official backbone default")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    parser.add_argument("--scheduler", choices=("poly", "cosine", "none"), default="poly")
    parser.add_argument("--lambda_aux", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--t_stab", type=int, default=2)
    parser.add_argument("--t_pert", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--num_perturb", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--aux_hidden_dim", type=int, default=256)
    parser.add_argument("--test_every", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run_root", default="runs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--pretrained", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size == 0:
        args.image_size = auto_image_size(args.backbone)
    method = "crsm" if args.use_crsm else "baseline"
    args.run_id = args.run_id or f"{args.backbone}_src{args.source_domain}_{method}_seed{args.seed}"
    run_dir = Path(args.run_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "console.log")
    config = vars(args).copy()
    config["selection_protocol"] = "target-domain oracle: best four-target mean DSC"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.test_every != 1:
        logger.warning("test_every=%d deviates from the requested every-epoch oracle protocol", args.test_every)
    seed_everything(args.seed)
    train_loader = make_loader(args.data_root, args.source_domain, args.image_size, args.batch_size, args.num_workers, args.seed, train=True)
    model = build_model(config).to(args.device)
    criterion = SegmentationLoss()
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=args.weight_decay, nesterov=True)
    if args.scheduler == "poly":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(0.0, 1.0 - epoch / max(args.epochs, 1)) ** 0.9)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    else:
        scheduler = None

    start_epoch, best_score, best_epoch = 1, -float("inf"), 0
    train_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    per_target_best: Dict[str, Dict[str, float | int]] = {}
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume["model"], strict=True)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = float(resume.get("selection_score", -float("inf")))
        best_epoch = int(resume.get("epoch", 0))
        train_rows = [row for row in read_history(run_dir / "train_metrics.csv") if int(row["epoch"]) < start_epoch]
        test_rows = [row for row in read_history(run_dir / "test_epoch_metrics.csv") if int(row["epoch"]) < start_epoch]
        for row in test_rows:
            domain = str(row.get("target_domain"))
            if domain == "TARGET_MEAN":
                continue
            old = per_target_best.get(domain)
            if old is None or float(row["mean_dice"]) > float(old["mean_dice"]):
                per_target_best[domain] = {"epoch": int(row["epoch"]), "mean_dice": float(row["mean_dice"])}
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    auxiliary_count = sum(parameter.numel() for name, parameter in model.named_parameters() if "stabilityAnalyzer" in name)
    logger.info("run=%s parameters=%d auxiliary_parameters=%d", args.run_id, parameter_count, auxiliary_count)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    training_started = time.perf_counter()
    best_path = run_dir / "best_target_mean.pth"
    gradient_audit_done = False

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {"weight": 0.0}
        for batch in train_loader:
            image = batch["image"].to(args.device, non_blocking=True)
            target = batch["mask"].to(args.device, non_blocking=True)
            output = model(image, compute_aux=args.use_crsm)
            logits = output["logits"]
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss_seg = criterion(logits, target)
            loss_aux = output["loss_aux"] if args.use_crsm else loss_seg.detach() * 0.0
            loss_total = loss_seg + args.lambda_aux * loss_aux
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            if not gradient_audit_done:
                encoder_grad = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for name, parameter in model.named_parameters() if "stabilityAnalyzer" not in name)
                auxiliary_grad = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for name, parameter in model.named_parameters() if "stabilityAnalyzer" in name)
                if not encoder_grad:
                    raise RuntimeError("Gradient audit failed: no finite gradient reached the segmentation model")
                if args.use_crsm and not auxiliary_grad:
                    raise RuntimeError("Gradient audit failed: no finite gradient reached the auxiliary network")
                if not args.use_crsm and hasattr(model, "stabilityAnalyzer"):
                    raise RuntimeError("Baseline audit failed: auxiliary module was constructed")
                logger.info("gradient_audit encoder=%s auxiliary=%s", encoder_grad, auxiliary_grad)
                gradient_audit_done = True
            optimizer.step()
            dice = train_dice(logits.detach(), target)
            batch_size = image.shape[0]
            append_mean(totals, {
                "loss_total": float(loss_total.detach()), "loss_seg": float(loss_seg.detach()),
                "loss_aux": float(loss_aux.detach()), "loss_pert": float(output["loss_pert"].detach()),
                "loss_stab": float(output["loss_stab"].detach()), "train_disc_dice": dice["disc"],
                "train_cup_dice": dice["cup"],
            }, batch_size)
        current_lr = optimizer.param_groups[0]["lr"]
        averages = mean_totals(totals)
        averages["train_mean_dice"] = (averages["train_disc_dice"] + averages["train_cup_dice"]) / 2.0
        train_row: Dict[str, object] = {"epoch": epoch, "learning_rate": current_lr, **averages}
        train_rows.append(train_row)
        if scheduler is not None:
            scheduler.step()
        logger.info(
            "epoch=%d lr=%.3e total=%.6f seg=%.6f aux=%.6f pert=%.6f stab=%.6f train_disc=%.4f train_cup=%.4f train_mean=%.4f",
            epoch, current_lr, averages["loss_total"], averages["loss_seg"], averages["loss_aux"],
            averages["loss_pert"], averages["loss_stab"], averages["train_disc_dice"],
            averages["train_cup_dice"], averages["train_mean_dice"],
        )
        if epoch % args.test_every == 0:
            epoch_domains, _ = evaluate_targets(
                model, args.source_domain, args.data_root, args.image_size, args.test_batch_size,
                args.num_workers, args.seed, args.device,
            )
            for row in epoch_domains:
                row["epoch"] = epoch
                test_rows.append(row)
                if row["target_domain"] != "TARGET_MEAN":
                    old = per_target_best.get(str(row["target_domain"]))
                    if old is None or float(row["mean_dice"]) > float(old["mean_dice"]):
                        per_target_best[str(row["target_domain"])] = {"epoch": epoch, "mean_dice": float(row["mean_dice"])}
                logger.info("test epoch=%d target=%s disc=%.4f cup=%.4f mean=%.4f disc_asd=%.4f cup_asd=%.4f",
                            epoch, row["target_domain"], row["disc_dice"], row["cup_dice"], row["mean_dice"], row["disc_asd"], row["cup_asd"])
            target_mean = next(row for row in epoch_domains if row["target_domain"] == "TARGET_MEAN")
            score = float(target_mean["mean_dice"])
            updated = score > best_score
            if updated:
                best_score, best_epoch = score, epoch
                checkpoint = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "selection_score": score, "config": config,
                }
                torch.save(checkpoint, best_path)
            logger.info("target_mean=%.4f current_best=%.4f best_epoch=%d checkpoint_updated=%s", score, best_score, best_epoch, updated)
        write_csv(run_dir / "train_metrics.csv", train_rows)
        write_csv(run_dir / "test_epoch_metrics.csv", test_rows)
        save_curves(run_dir / "curves.png", train_rows, test_rows, best_epoch)

    elapsed = time.perf_counter() - training_started
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was produced; ensure test_every permits at least one evaluation")
    best_checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model"], strict=True)
    metadata = {
        "run_id": args.run_id, "method": method, "backbone": args.backbone,
        "source_domain": args.source_domain, "seed": args.seed, "best_epoch": best_epoch,
    }
    best_domains, best_cases = evaluate_targets(
        model, args.source_domain, args.data_root, args.image_size, args.test_batch_size,
        args.num_workers, args.seed, args.device, metadata,
    )
    write_csv(run_dir / "test_case_metrics_best.csv", best_cases)
    write_csv(run_dir / "test_domain_metrics_best.csv", best_domains)
    peak_memory_mb = 0.0
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)
    summary = {
        "protocol": "target-domain oracle model selection using four-target mean DSC",
        "best_epoch": best_epoch, "best_target_mean_dice": best_score,
        "per_target_historical_best": per_target_best,
        "per_target_checkpoints_saved": False,
        "domains_at_selected_checkpoint": best_domains,
        "parameter_count": parameter_count, "auxiliary_parameter_count": auxiliary_count,
        "training_seconds": elapsed, "peak_memory_mb": peak_memory_mb, "config": config,
    }
    (run_dir / "test_best_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("finished best_epoch=%d best_target_mean=%.4f seconds=%.1f", best_epoch, best_score, elapsed)


if __name__ == "__main__":
    main()
