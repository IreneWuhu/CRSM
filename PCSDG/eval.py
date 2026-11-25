"""Evaluate a saved final/last PCSDG+CRSM checkpoint on all non-source domains."""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from data_optic import DOMAINS as OD_DOMAINS, OpticDataset
from data_polyp import DOMAINS as POLYP_DOMAINS, PolypDataset
from pcsdg_crsm import PCSDGCRSM
from pcsdg_augment import normalize_tensor
from train import dice_rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--dataset", choices=("odoc", "polyp"), required=True)
    p.add_argument("--data_root", required=True); p.add_argument("--output_dir", default=None)
    p.add_argument("--batch_size", type=int, default=1); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0"); a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() or "cuda" not in a.device else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    cfg = ckpt["config"]; domains = OD_DOMAINS if a.dataset == "odoc" else POLYP_DOMAINS
    model = PCSDGCRSM(task=a.dataset, backbone=cfg["backbone"], pretrained=False,
        use_crsm=cfg["use_crsm"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True); model.eval()
    out = Path(a.output_dir or (Path(a.checkpoint).parent / "evaluation")); out.mkdir(parents=True, exist_ok=True)
    case_rows, domain_rows = [], []
    with torch.no_grad():
        for domain in domains:
            if domain == cfg["source_domain"]: continue
            ds = OpticDataset(a.data_root, domain, cfg["image_size"], False) if a.dataset == "odoc" else PolypDataset(a.data_root, domain, cfg["image_size"], False)
            loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
            rows = []
            for batch in loader:
                image = normalize_tensor(batch["image"].to(device))
                output = model(image, compute_aux=False, return_embeddings=False)
                rows.extend(dice_rows(output["logits"], batch["mask"].to(device), batch["case_id"], a.dataset, int(ckpt["epoch"]), domain))
            case_rows.extend(rows)
            domain_rows.append({"domain": domain, "n_cases": len(rows), "mean_dsc": float(np.mean([r["mean_dsc"] for r in rows])),
                                "std_case_dsc": float(np.std([r["mean_dsc"] for r in rows]))})
    for name, rows in (("test_case_metrics.csv", case_rows), ("test_domain_metrics.csv", domain_rows)):
        if rows:
            with open(out / name, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(domain_rows)

if __name__ == "__main__": main()
