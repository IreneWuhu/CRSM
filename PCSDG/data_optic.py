"""VPTTA-compatible OD/OC CSV loader for the five fundus domains."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DOMAINS: Dict[str, str] = {
    "A": "RIM_ONE_r3", "B": "REFUGE", "C": "ORIGA",
    "D": "REFUGE_Valid", "E": "Drishti_GS",
}


def csv_names(domain: str) -> List[str]:
    name = DOMAINS[domain]
    return [f"{name}.csv"] if name == "REFUGE_Valid" else [f"{name}_train.csv", f"{name}_test.csv"]


def read_pairs(root: Path, names: Sequence[str]) -> List[Tuple[str, str]]:
    pairs = []
    for name in names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"image", "mask"}.issubset(reader.fieldnames):
                raise ValueError(f"{path} must contain image and mask columns")
            pairs.extend((row["image"], row["mask"]) for row in reader)
    return pairs


class OpticDataset(Dataset):
    def __init__(self, root: str, domain: str, image_size=512, train=False):
        self.root = Path(root)
        self.domain = domain
        self.image_size = image_size
        self.train = train
        self.samples = read_pairs(self.root, csv_names(domain))

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if random.random() < 0.1:
            angle = random.uniform(-15, 15)
            scale = random.uniform(0.75, 1.25)
            image = TF.affine(image, angle, (0, 0), scale, 0, InterpolationMode.BICUBIC, fill=0)
            mask = TF.affine(mask, angle, (0, 0), scale, 0, InterpolationMode.NEAREST, fill=255)
        return image, mask

    def __getitem__(self, index):
        image_rel, mask_rel = self.samples[index]
        if mask_rel.lower().endswith(".tif"):
            mask_rel = mask_rel[:-4] + "-1.tif"
        image_path, mask_path = self.root / image_rel, self.root / mask_rel
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        if self.train:
            image, mask = self._augment(image, mask)
        image_np = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        # Keep the 0..255 intensity range until PCSDG's per-image z-score.
        # PCSDG's SABA collate omits multiplicative brightness and uses a small
        # Gaussian noise variance (0, 0.05) before per-channel z-score.
        if self.train and random.random() < 0.15:
            image_np = np.clip(image_np + np.random.normal(0, random.uniform(0, np.sqrt(0.05)), image_np.shape), 0, 255)
        image_np = image_np.astype(np.float32, copy=False)
        label = np.asarray(mask)
        mask_np = np.stack([label < 255, label == 0], axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(image_np),
            "mask": torch.from_numpy(mask_np),
            "case_id": Path(image_rel).as_posix(),
        }


def seed_worker(worker_id):
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def make_loader(root, domain, image_size, batch_size, workers, seed, train):
    dataset = OpticDataset(root, domain, image_size, train)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=False,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, generator=generator)


def target_domains(source: str):
    return [domain for domain in DOMAINS if domain != source]
