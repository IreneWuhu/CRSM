"""OD/OC five-domain CSV data interface compatible with the VPTTA layout."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DOMAIN_NAMES: Dict[str, str] = {
    "A": "RIM_ONE_r3",
    "B": "REFUGE",
    "C": "ORIGA",
    "D": "REFUGE_Valid",
    "E": "Drishti_GS",
}


def domain_csvs(domain: str) -> List[str]:
    name = DOMAIN_NAMES[domain]
    return [f"{name}.csv"] if name == "REFUGE_Valid" else [f"{name}_train.csv", f"{name}_test.csv"]


def read_csv_pairs(root: Path, csv_names: Sequence[str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for csv_name in csv_names:
        csv_path = root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing OD/OC split file: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"image", "mask"}.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} must contain image and mask columns")
            rows.extend((str(row["image"]), str(row["mask"])) for row in reader)
    return rows


def _mask_to_two_channels(label: np.ndarray) -> np.ndarray:
    # Preserve the exact VPTTA convention: <255 is disc, 0 is cup.
    disc = label < 255
    cup = label == 0
    return np.stack([disc, cup], axis=0).astype(np.float32)


class OpticDataset(Dataset):
    def __init__(self, root: str | Path, domain: str, image_size: int, augment: bool = False):
        if domain not in DOMAIN_NAMES:
            raise ValueError(f"Unknown domain {domain}; choose one of {sorted(DOMAIN_NAMES)}")
        self.root = Path(root)
        self.domain = domain
        self.image_size = int(image_size)
        self.augment = augment
        self.samples = read_csv_pairs(self.root, domain_csvs(domain))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image, label: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image, label = TF.hflip(image), TF.hflip(label)
        if random.random() < 0.5:
            image, label = TF.vflip(image), TF.vflip(label)
        if random.random() < 0.1:
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.75, 1.25)
            image = TF.affine(image, angle, (0, 0), scale, 0.0, InterpolationMode.BICUBIC, fill=0)
            label = TF.affine(label, angle, (0, 0), scale, 0.0, InterpolationMode.NEAREST, fill=255)
        if random.random() < 0.15:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.5))
        return image, label

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_rel, mask_rel = self.samples[index]
        if mask_rel.lower().endswith(".tif"):
            mask_rel = mask_rel[:-4] + "-1.tif"
        image_path = self.root / image_rel
        mask_path = self.root / mask_rel
        image = Image.open(image_path).convert("RGB")
        label = Image.open(mask_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        label = label.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        if self.augment:
            image, label = self._augment(image, label)
        image_np = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        lo, hi = float(image_np.min()), float(image_np.max())
        image_np = (image_np - lo) / max(hi - lo, 1e-8)
        if self.augment and random.random() < 0.15:
            image_np = np.clip(image_np + np.random.normal(0.0, random.uniform(0.0, 0.05), image_np.shape), 0, 1)
        mask_np = _mask_to_two_channels(np.asarray(label))
        # Keep the relative path (not only the stem) so paired statistics cannot
        # silently merge identically named images from different subfolders.
        case_id = Path(image_rel).as_posix()
        return {
            "image": torch.from_numpy(image_np.astype(np.float32)),
            "mask": torch.from_numpy(mask_np),
            "case_id": case_id,
            "image_path": str(image_path),
            "domain": self.domain,
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    root: str | Path,
    domain: str,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    train: bool,
) -> DataLoader:
    dataset = OpticDataset(root, domain, image_size=image_size, augment=train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def target_domains(source_domain: str) -> List[str]:
    return [domain for domain in DOMAIN_NAMES if domain != source_domain]
