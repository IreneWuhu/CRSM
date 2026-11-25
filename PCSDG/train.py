"""Single-source PCSDG + CRSM training for OD/OC and Polyp."""
from __future__ import annotations
import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from data_optic import DOMAINS as OD_DOMAINS, OpticDataset
from data_polyp import DOMAINS as POLYP_DOMAINS, PolypDataset
from pcsdg_crsm import PCSDGCRSM, pcsdg_pair_losses
from pcsdg_augment import make_views, normalize_tensor

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_loader(dataset, batch_size, workers, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        drop_last=False)

def seg_targets(mask, task):
    if task == "odoc": return mask
    return mask[:, :1]

def seg_loss(logits, mask, task):
    target = seg_targets(mask, task)
    return sum(F.binary_cross_entropy_with_logits(logits[:, i], target[:, i])
               for i in range(logits.shape[1]))

def dice_rows(logits, mask, case_ids, task, epoch, domain):
    pred = torch.sigmoid(logits) >= .5; target = seg_targets(mask, task) >= .5
    rows = []
    for i, case_id in enumerate(case_ids):
        scores = []
        for c in range(pred.shape[1]):
            p, g = pred[i, c], target[i, c]
            score = 100 * (2 * (p & g).sum().item() + 1e-6) / (p.sum().item() + g.sum().item() + 1e-6)
            scores.append(score)
        row = {"epoch": epoch, "domain": domain, "case_id": case_id}
        if task == "odoc": row.update(disc_dsc=scores[0], cup_dsc=scores[1], mean_dsc=sum(scores)/2)
        else: row.update(dsc=scores[0], mean_dsc=scores[0])
        rows.append(row)
    return rows

def write_rows(path, rows):
    if not rows: return
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists: writer.writeheader()
        writer.writerows(rows)

def discard_incomplete_rows(path, completed_epoch):
    path = Path(path)
    if not path.exists(): return
    with path.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src); fields = reader.fieldnames
        kept = [row for row in reader if int(row["epoch"]) <= completed_epoch]
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields); writer.writeheader(); writer.writerows(kept)

def save_curves(out):
    train_rows = list(csv.DictReader(open(out / "train_metrics.csv", encoding="utf-8")))
    test_rows = list(csv.DictReader(open(out / "test_epoch_metrics.csv", encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [int(r["epoch"]) for r in train_rows]
    axes[0].plot(epochs, [float(r["loss_seg"]) for r in train_rows], label="seg")
    axes[0].plot(epochs, [float(r["loss_pc_content"]) for r in train_rows], label="PCSDG content")
    axes[0].plot(epochs, [float(r["loss_pc_style"]) for r in train_rows], label="PCSDG style")
    axes[0].plot(epochs, [float(r["loss_aux"]) for r in train_rows], label="CRSM")
    axes[0].set_title("Training losses"); axes[0].legend(); axes[0].grid(alpha=.25)
    for domain in sorted({r["domain"] for r in test_rows}):
        rows = [r for r in test_rows if r["domain"] == domain]
        axes[1].plot([int(r["epoch"]) for r in rows], [float(r["mean_dsc"]) for r in rows], label=domain)
    axes[1].set_title("Target-domain DSC"); axes[1].legend(); axes[1].grid(alpha=.25)
    for ax in axes: ax.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=140); plt.close(fig)

def evaluate(model, root, task, source, domains, image_size, device, epoch, out):
    model.eval(); per_case = []; domain_rows = []
    with torch.no_grad():
        for domain in domains:
            ds = (OpticDataset(root, domain, image_size, train=False) if task == "odoc"
                  else PolypDataset(root, domain, image_size, train=False))
            loader = make_loader(ds, 1, 0, False)
            rows = []
            for batch in loader:
                image, mask = normalize_tensor(batch["image"].to(device)), batch["mask"].to(device)
                output = model(image, compute_aux=False, return_embeddings=False)
                rows += dice_rows(output["logits"], mask, batch["case_id"], task, epoch, domain)
            per_case.extend(rows)
            values = [r["mean_dsc"] for r in rows]
            summary = {"epoch": epoch, "domain": domain, "n_cases": len(rows),
                       "mean_dsc": float(np.mean(values)), "std_case_dsc": float(np.std(values))}
            if task == "odoc":
                summary["disc_dsc"] = float(np.mean([r["disc_dsc"] for r in rows]))
                summary["cup_dsc"] = float(np.mean([r["cup_dsc"] for r in rows]))
            domain_rows.append(summary)
    write_rows(out / "test_case_metrics.csv", per_case)
    write_rows(out / "test_epoch_metrics.csv", domain_rows)
    return domain_rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("odoc", "polyp"), required=True)
    p.add_argument("--source_domain", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_root", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--run_root", default="runs")
    p.add_argument("--backbone", default=None)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100, help="maximum epochs and LR schedule horizon")
    p.add_argument("--lr", type=float, default=1e-3, help="main and PCSDG-module AdamW initial LR")
    p.add_argument("--u_lr", type=float, default=1e-4, help="CRSM U/BN AdamW initial LR")
    p.add_argument("--min_epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=None,
                   help="required smoothed improvement; defaults to 0.1 DSC points or 0.001 loss")
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--stop_metric", choices=("target_mean", "source_seg_loss"), default="target_mean")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--lambda_crsm", type=float, default=.01)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use-crsm", dest="use_crsm", action="store_true")
    p.add_argument("--no-use-crsm", dest="use_crsm", action="store_false"); p.set_defaults(use_crsm=True)
    p.add_argument("--pretrained", dest="pretrained", action="store_true")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false"); p.set_defaults(pretrained=True)
    p.add_argument("--resume", default=None)
    a = p.parse_args()
    if a.min_delta is None:
        a.min_delta = .1 if a.stop_metric == "target_mean" else .001
    if a.epochs < a.min_epochs or a.min_epochs < a.smooth_window or a.patience < 1:
        p.error("require epochs >= min_epochs >= smooth_window and patience >= 1")
    if a.min_delta < 0:
        p.error("min_delta must be nonnegative")
    all_domains = OD_DOMAINS if a.dataset == "odoc" else POLYP_DOMAINS
    if a.source_domain not in all_domains: p.error("source domain invalid for dataset")
    if a.backbone is None: a.backbone = "resnet34" if a.dataset == "odoc" else "resnet50"
    seed_all(a.seed); device = torch.device(a.device if torch.cuda.is_available() or "cuda" not in a.device else "cpu")
    out = Path(a.run_root) / a.run_id; out.mkdir(parents=True, exist_ok=True)
    config = vars(a).copy(); config["device"] = str(device)
    config["main_prompt_optimizer"] = "AdamW lr=%g, betas=(0.9,0.99), wd=%g; main polynomial decay, prompt fixed LR" % (a.lr, a.weight_decay)
    config["crsm_optimizer"] = ("AdamW lr=%g, betas=(0.9,0.99), wd=5e-4, polynomial decay" % a.u_lr
                                if a.use_crsm else None)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_ds = (OpticDataset(a.data_root, a.source_domain, a.image_size, train=True) if a.dataset == "odoc"
                else PolypDataset(a.data_root, a.source_domain, a.image_size, train=True))
    train_loader = make_loader(train_ds, a.batch_size, a.workers, True)
    model = PCSDGCRSM(task=a.dataset, backbone=a.backbone, pretrained=a.pretrained,
        use_crsm=a.use_crsm).to(device)
    # Preserve the two PCSDG update roles, but use the requested AdamW for both.
    prompt_params = list(model.dwm.parameters()) + list(model.projector.parameters())
    excluded_ids = {id(t) for t in model.projector.parameters()}
    if a.use_crsm: excluded_ids.update(id(t) for t in model.stabilityAnalyzer.parameters())
    main_params = [t for t in model.parameters() if id(t) not in excluded_ids]
    opt_main = torch.optim.AdamW(main_params, lr=a.lr, betas=(.9, .99), weight_decay=a.weight_decay)
    opt_prompt = torch.optim.AdamW(prompt_params, lr=a.lr, betas=(.9, .99), weight_decay=a.weight_decay)
    # CRSM U/BN stays on its separate, lower-LR AdamW optimizer.
    opt_crsm = (torch.optim.AdamW(model.stabilityAnalyzer.parameters(), lr=a.u_lr,
                                  betas=(.9, .99), weight_decay=5e-4) if a.use_crsm else None)
    start_epoch = 0
    stop_state = {"values": [], "best_smoothed": None, "stale_epochs": 0}
    if a.resume:
        state = torch.load(a.resume, map_location=device)
        model.load_state_dict(state["model"]); opt_main.load_state_dict(state["opt_main"])
        opt_prompt.load_state_dict(state["opt_prompt"]); start_epoch = int(state["epoch"])
        if opt_crsm is not None and state.get("opt_crsm") is not None:
            opt_crsm.load_state_dict(state["opt_crsm"])
        stop_state = state.get("stop_state", stop_state)
    if start_epoch == 0:
        (out / "console.log").write_text("", encoding="utf-8")
    log = open(out / "console.log", "a", encoding="utf-8", buffering=1)
    train_csv, test_csv = out / "train_metrics.csv", out / "test_epoch_metrics.csv"
    if start_epoch == 0:
        for path in (train_csv, test_csv, out / "test_case_metrics.csv"):
            if path.exists(): path.unlink()
    else:
        for path in (train_csv, test_csv, out / "test_case_metrics.csv"):
            discard_incomplete_rows(path, start_epoch)
    completed_epoch = start_epoch
    stop_reason = "maximum_epochs"
    test_domains = [d for d in all_domains if d != a.source_domain]
    for epoch in range(start_epoch, a.epochs):
        tic = time.time(); lr = a.lr * (1 - epoch / a.epochs) ** .9
        for group in opt_main.param_groups: group["lr"] = lr
        if opt_crsm is not None:
            crsm_lr = a.u_lr * (1 - epoch / a.epochs) ** .9
            for group in opt_crsm.param_groups: group["lr"] = crsm_lr
        model.train(); sums = {k: [] for k in ("loss_total", "loss_seg", "loss_pc_content", "loss_pc_style", "loss_aux", "loss_pert", "loss_stab")}
        for batch in train_loader:
            x_cpu, y_cpu = batch["image"], batch["mask"]
            views = [v.to(device, non_blocking=True) for v in make_views(x_cpu, y_cpu)]
            y = y_cpu.to(device, non_blocking=True)
            # PCSDG pixel-level content invariance / style separation update.
            opt_prompt.zero_grad(set_to_none=True)
            embs = [model.get_disentangled_embeddings(view) for view in views]
            content_loss, style_loss = pcsdg_pair_losses(embs)
            (content_loss + style_loss).backward(); opt_prompt.step()
            # PCSDG supervises FDA, GLA, SABA, and original source views.
            batch_seg = []; batch_total = []; aux_values = None
            for view_index, view in enumerate((views[1], views[2], views[3], views[0])):
                opt_main.zero_grad(set_to_none=True)
                if opt_crsm is not None: opt_crsm.zero_grad(set_to_none=True)
                result = model(view, compute_aux=(a.use_crsm and view_index == 3), return_embeddings=False)
                segmentation = seg_loss(result["logits"], y, a.dataset)
                loss = segmentation
                if view_index == 3 and a.use_crsm:
                    loss = loss + a.lambda_crsm * result["loss_aux"]
                    aux_values = result
                loss.backward(); opt_main.step()
                if opt_crsm is not None and view_index == 3: opt_crsm.step()
                batch_seg.append(float(segmentation.detach()))
                batch_total.append(float(loss.detach()))
            sums["loss_seg"].append(float(np.mean(batch_seg)))
            sums["loss_total"].append(float(np.mean(batch_total)))
            sums["loss_pc_content"].append(float(content_loss.detach()))
            sums["loss_pc_style"].append(float(style_loss.detach()))
            for key, name in (("loss_aux", "loss_aux"), ("loss_pert", "loss_pert"), ("loss_stab", "loss_stab")):
                sums[key].append(float(aux_values[name].detach()) if aux_values is not None else 0.)
        train_row = {"epoch": epoch + 1, "lr": lr, "n_train": len(train_ds),
                     **{k: float(np.mean(v)) for k, v in sums.items()},
                     "epoch_seconds": time.time() - tic}
        domain_rows = evaluate(model, a.data_root, a.dataset, a.source_domain, test_domains,
                               a.image_size, device, epoch + 1, out)
        avg = float(np.mean([r["mean_dsc"] for r in domain_rows]))
        current = avg if a.stop_metric == "target_mean" else train_row["loss_seg"]
        stop_state["values"].append(current)
        recent = stop_state["values"][-a.smooth_window:]
        smoothed = float(np.mean(recent))
        if epoch + 1 == a.min_epochs:
            stop_state["best_smoothed"] = smoothed
            stop_state["stale_epochs"] = 0
        elif epoch + 1 > a.min_epochs:
            best = stop_state["best_smoothed"]
            improvement = (smoothed - best if a.stop_metric == "target_mean"
                           else best - smoothed)
            if improvement >= a.min_delta:
                stop_state["best_smoothed"] = smoothed
                stop_state["stale_epochs"] = 0
            else:
                stop_state["stale_epochs"] += 1
        train_row.update(target_mean_dsc=avg, stop_score_smoothed=smoothed,
                         stop_stale_epochs=stop_state["stale_epochs"])
        write_rows(train_csv, [train_row])
        # Keep latest fully resumable state; no single-epoch best checkpoint is selected.
        tmp_checkpoint = out / "last.pth.tmp"
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "opt_main": opt_main.state_dict(),
                    "opt_prompt": opt_prompt.state_dict(),
                    "opt_crsm": opt_crsm.state_dict() if opt_crsm is not None else None,
                    "stop_state": stop_state, "config": config}, tmp_checkpoint)
        os.replace(tmp_checkpoint, out / "last.pth")
        completed_epoch = epoch + 1
        msg = "epoch=%d lr=%.6g seg=%.5f pc_content=%.5f pc_style=%.5f aux=%.6g pert=%.6g stab=%.6g target_mean=%.3f stop_smoothed=%.3f stale=%d sec=%.1f" % (
            epoch+1, lr, train_row["loss_seg"], train_row["loss_pc_content"], train_row["loss_pc_style"],
            train_row["loss_aux"], train_row["loss_pert"], train_row["loss_stab"], avg,
            smoothed, stop_state["stale_epochs"], train_row["epoch_seconds"])
        print(msg, flush=True); log.write(msg + "\n")
        if completed_epoch >= a.min_epochs and stop_state["stale_epochs"] >= a.patience:
            stop_reason = "patience"
            break
    torch.save({"epoch": completed_epoch, "model": model.state_dict(), "config": config}, out / "final.pth")
    rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
    last = [r for r in rows if int(r["epoch"]) > max(0, completed_epoch - 5)]
    by_domain = {}
    for domain in test_domains:
        vals = [float(r["mean_dsc"]) for r in last if r["domain"] == domain]
        by_domain[domain] = {"last5_mean_dsc": float(np.mean(vals)),
                             "last5_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.,
                             "n_epochs": len(vals)}
    epoch_avgs = [float(np.mean([float(r["mean_dsc"]) for r in last if int(r["epoch"]) == e]))
                  for e in range(max(1, completed_epoch - 4), completed_epoch + 1)]
    summary = {"config": config, "final_epoch": completed_epoch, "stop_reason": stop_reason,
               "stopping": {"metric": a.stop_metric, "smoothing_window": a.smooth_window,
                            "min_epochs": a.min_epochs, "patience": a.patience,
                            "min_delta": a.min_delta},
               "final_epoch_target_mean_dsc": stop_state["values"][-1] if a.stop_metric == "target_mean" else float(np.mean([float(r["mean_dsc"]) for r in rows if int(r["epoch"]) == completed_epoch])),
               "last5_target_mean_dsc": float(np.mean(epoch_avgs)),
               "last5_target_sd": float(np.std(epoch_avgs, ddof=1)) if len(epoch_avgs) > 1 else 0.,
               "target_domains_last5": by_domain,
               "selection": "final checkpoint at stopping epoch; no best-epoch selection"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_curves(out)
    log.close()

if __name__ == "__main__": main()
