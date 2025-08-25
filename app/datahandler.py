# sunsets/model/datahandler.py
from __future__ import annotations

# Python imports
import json
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import sys

# 3rd-party imports
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# 1st-party imports
BASE_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
# print("REPO_DIR:", BASE_DIR)
from app.config import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

Transform = Callable[[Image.Image], Tensor]


def create_transform(image_size: int) -> Transform:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _parse_score_to_index(raw_score: object, num_classes: int) -> Optional[int]:
    """Parse raw JSON 'score' into a 0-based class index or return None if invalid."""
    if raw_score is None:
        return None
    s = str(raw_score).strip()
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        score_int = int(m.group())
    except Exception:
        return None
    if 1 <= score_int <= num_classes:
        return score_int - 1
    return None


def _image_is_ok(path: Path, verify_images: bool = True) -> bool:
    """Check that path exists and (optionally) is a readable image."""
    if not path.is_file():
        return False
    if not verify_images:
        return True
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class SunsetDataset(Dataset[Tuple[Tuple[Tensor, Tensor], int]]):
    """Dataset for paired sunset images (-2h and -1h) with score label.

    Skips rows that are incomplete or malformed.
    """

    def __init__(
        self,
        json_path: Path,
        image_dir: Path,
        transform: Optional[Transform] = None,
        verify_images: bool = True,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.image_dir: Path = image_dir
        self.transform: Optional[Transform] = transform
        self.num_classes: int = num_classes

        with json_path.open("r", encoding="utf-8") as f:
            data: List[Dict] = json.load(f)

        self.samples: List[Tuple[Path, Path, int]] = []
        skipped_missing_types = 0
        skipped_bad_score = 0
        skipped_bad_image = 0
        total_rows = 0

        for entry in data:
            total_rows += 1
            images: List[Dict[str, str]] = entry.get("images", []) or []
            img_dict: Dict[str, str] = {
                (img_info.get("type") or ""): (img_info.get("name") or "")
                for img_info in images
                if isinstance(img_info, dict)
            }
            if "-2h" not in img_dict or "-1h" not in img_dict:
                skipped_missing_types += 1
                continue

            score_idx = _parse_score_to_index(entry.get("score"), self.num_classes)
            if score_idx is None:
                skipped_bad_score += 1
                continue

            path_neg2 = image_dir / img_dict["-2h"]
            path_neg1 = image_dir / img_dict["-1h"]

            if not (
                _image_is_ok(path_neg2, verify_images)
                and _image_is_ok(path_neg1, verify_images)
            ):
                skipped_bad_image += 1
                continue

            self.samples.append((path_neg2, path_neg1, score_idx))

        if skipped_missing_types or skipped_bad_score or skipped_bad_image:
            warnings.warn(
                "[SunsetDataset] Loaded "
                f"{len(self.samples)}/{total_rows} valid pairs. "
                f"Skipped: missing types={skipped_missing_types}, "
                f"bad score={skipped_bad_score}, bad image={skipped_bad_image}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Tensor:
        img = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return transforms.ToTensor()(img)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], int]:
        img_path_2h, img_path_1h, score = self.samples[idx]
        img_2h = self._load_image(img_path_2h)
        img_1h = self._load_image(img_path_1h)
        return (img_2h, img_1h), int(score)


def create_dataloaders(
    json_path: Path,
    image_dir: Path,
    transform: Transform,
    batch_size: int,
    val_split: float,
    seed: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    dataset = SunsetDataset(
        json_path=json_path,
        image_dir=image_dir,
        transform=transform,
        verify_images=True,
        num_classes=NUM_CLASSES,
    )

    if len(dataset) == 0:
        empty_loader = DataLoader(dataset, batch_size=batch_size)
        warnings.warn(
            "[create_dataloaders] Dataset is empty after cleaning; training will be a no-op."
        )
        return empty_loader, empty_loader

    val_size = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    cpu_count = max(1, (torch.get_num_threads() or 1))
    num_workers = min(4, cpu_count) - 1
    num_workers = max(0, num_workers)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader
