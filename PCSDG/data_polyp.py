"""Polyp CSV adapter compatible with the VPTTA image/mask list convention."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random

DOMAINS = {"A": "BKAI", "B": "CVC-ClinicDB", "C": "ETIS-LaribPolypDB", "D": "Kvasir-SEG"}

def _read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header: %s" % path)
        fields = {v.lower(): v for v in reader.fieldnames}
        image_key = next((fields[k] for k in ("image", "img", "image_path") if k in fields), None)
        mask_key = next((fields[k] for k in ("mask", "label", "gt", "mask_path") if k in fields), None)
        if not image_key or not mask_key:
            raise ValueError("CSV needs image and mask columns: %s" % path)
        return [(row[image_key].strip(), row[mask_key].strip()) for row in reader if row.get(image_key) and row.get(mask_key)]

def find_csvs(root: Path, domain: str):
    name = DOMAINS[domain]
    direct = [root / (name + "_train.csv"), root / (name + "_test.csv")]
    if all(p.is_file() for p in direct):
        return direct
    for candidate in (root / (name + ".csv"), root / name / "train.csv", root / name / "all.csv"):
        if candidate.is_file():
            return [candidate]
    hits = sorted(p for p in root.rglob("*.csv") if name.lower().replace("-", "") in p.as_posix().lower().replace("-", ""))
    if hits:
        return hits
    raise FileNotFoundError("No Polyp CSV found for %s (%s); expected CSV fields image,mask" % (domain, name))

class PolypDataset(Dataset):
    def __init__(self, root, domain, image_size=512, train=False):
        self.root, self.domain, self.image_size, self.train = Path(root), domain, image_size, train
        self.rows = [pair for csv_path in find_csvs(self.root, domain) for pair in _read_rows(csv_path)]
        if not self.rows:
            raise ValueError("No rows found for Polyp domain %s" % domain)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        image_rel, mask_rel = self.rows[index]
        ip, mp = Path(image_rel), Path(mask_rel)
        if not ip.is_absolute(): ip = self.root / ip
        if not mp.is_absolute(): mp = self.root / mp
        image = Image.open(ip).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(mp).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        if self.train and random.random() < .5:
            image, mask = image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        # PCSDG's official collate path applies per-image/channel z-score after augmentation.
        x = np.asarray(image, dtype=np.float32)
        y = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)[None]
        try: case_id = ip.relative_to(self.root).as_posix()
        except ValueError: case_id = str(ip)
        return {"image": torch.from_numpy(x.transpose(2, 0, 1).copy()), "mask": torch.from_numpy(y),
                "case_id": case_id}

def make_loader(root, domain, image_size, batch_size, workers, seed, train):
    ds = PolypDataset(root, domain, image_size, train)
    gen = torch.Generator().manual_seed(seed)
    def init_worker(worker_id):
        value = torch.initial_seed() % (2**32); random.seed(value); np.random.seed(value)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, worker_init_fn=init_worker,
                      generator=gen, drop_last=False)

def target_domains(source): return [key for key in DOMAINS if key != source]
