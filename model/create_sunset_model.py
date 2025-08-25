# sunsets/model/create_sunset_model.py
from __future__ import annotations

import json
import random
import re  # NEW: for robust score parsing
import sys
import warnings  # NEW: lightweight logging for skipped samples
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt  # noqa: F401  (kept for potential future use)


# -----------------------------
# Paths & Constants (relative to the "sunsets" folder)
# -----------------------------
# This file lives at: sunsets/model/create_sunset_model.py
BASE_DIR: Path = Path(__file__).resolve().parent.parent  # -> sunsets/

JSON_PATH: Path = BASE_DIR / "sunset_images_grouped.json"
IMAGE_DIR: Path = BASE_DIR / "img" / "ob_hotel"
MODEL_DIR: Path = BASE_DIR / "model"
MODEL_PATH: Path = MODEL_DIR / "sunset_dual_resnet18_best.pth"

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Five classes: 1,2,3,4,5 in the JSON (we map to 0..4 internally)
NUM_CLASSES: int = 5
IMAGE_SIZE: int = 64
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-4
VAL_SPLIT: float = 0.2
RANDOM_SEED: int = 42

# ImageNet mean & std for normalization
IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)

Transform = Callable[[Image.Image], Tensor]


def safe_load_state_dict(
    model: nn.Module, ckpt_path: Path, device: torch.device
) -> nn.Module:
    """Load *compatible* weights from ``ckpt_path`` into ``model``.

    Layers whose tensor shapes don’t match (e.g., the final classifier when NUM_CLASSES
    has changed) are skipped instead of raising a RuntimeError.
    """
    state = torch.load(ckpt_path, map_location=device)
    own_state = model.state_dict()

    # keep only parameters whose shape matches the current model
    filtered = {
        k: v
        for k, v in state.items()
        if k in own_state and v.shape == own_state[k].shape
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(
            f"[safe_load_state_dict] skipped keys – missing: {missing}  unexpected: {unexpected}"
        )
    return model


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utilities for data hygiene
# -----------------------------
def _parse_score_to_index(raw_score: object, num_classes: int) -> Optional[int]:
    """Parse raw JSON 'score' into a 0-based class index or return None if invalid.

    Accepts strings like '3', ' 5 ', 'score: 2' and negative/garbage will be rejected.
    """
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
        return score_int - 1  # shift to 0-based
    return None


def _image_is_ok(path: Path, verify_images: bool = True) -> bool:
    """Check that path exists and (optionally) is a readable image."""
    if not path.is_file():
        return False
    if not verify_images:
        return True
    try:
        # verify() is a lightweight consistency check; we reopen for actual load later
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class SunsetDataset(Dataset[Tuple[Tuple[Tensor, Tensor], int]]):
    """PyTorch Dataset for paired sunset images (-2h and -1h) with score label.

    This dataset *skips* malformed/incomplete rows:
      - missing '-2h' or '-1h'
      - missing/non-integer score, or score outside [1..NUM_CLASSES]
      - image files that don't exist or fail a quick PIL verify()
    """

    def __init__(
        self,
        json_path: Path,
        image_dir: Path,
        transform: Optional[Transform] = None,
        verify_images: bool = True,  # NEW: opt-in file integrity check
    ) -> None:
        self.image_dir: Path = image_dir
        self.transform: Optional[Transform] = transform

        with json_path.open("r", encoding="utf-8") as f:
            data: List[Dict] = json.load(f)

        # Build samples: [(img_path_-2h, img_path_-1h, score_idx), ...]
        self.samples: List[Tuple[Path, Path, int]] = []
        skipped_missing_types = 0
        skipped_bad_score = 0
        skipped_bad_image = 0
        total_rows = 0

        for entry in data:
            total_rows += 1
            images: List[Dict[str, str]] = entry.get("images", []) or []
            img_dict: Dict[str, str] = {
                img_info.get("type", ""): img_info.get("name", "")
                for img_info in images
                if isinstance(img_info, dict)
            }

            # need both -2h and -1h
            if "-2h" not in img_dict or "-1h" not in img_dict:
                skipped_missing_types += 1
                continue

            score_idx = _parse_score_to_index(entry.get("score"), NUM_CLASSES)
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
        # robust load; if this fails despite verify(), let the exception bubble up
        img = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return transforms.ToTensor()(img)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], int]:
        img_path_2h, img_path_1h, score = self.samples[idx]
        img_2h = self._load_image(img_path_2h)
        img_1h = self._load_image(img_path_1h)
        return (img_2h, img_1h), int(score)  # ensure Python int for default collate


class DualResNet(nn.Module):
    """Dual-branch ResNet18 model that processes two images and outputs class logits."""

    def __init__(
        self, num_classes: int = NUM_CLASSES, shared_backbone: bool = True
    ) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )  # Output: (B, 512, 1, 1)

        # If not sharing weights, create a second backbone
        if shared_backbone:
            self.feature_extractor2 = self.feature_extractor
        else:
            backbone2 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_extractor2 = nn.Sequential(*list(backbone2.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img_2h: Tensor, img_1h: Tensor) -> Tensor:
        f1 = self.feature_extractor(img_2h).flatten(1)
        f2 = self.feature_extractor2(img_1h).flatten(1)
        concatenated = torch.cat([f1, f2], dim=1)
        logits = self.classifier(concatenated)
        return logits


# -----------------------------
# Training & Evaluation Helpers
# -----------------------------
@dataclass(frozen=True)
class TrainConfig:
    image_size: int = IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    val_split: float = VAL_SPLIT
    num_classes: int = NUM_CLASSES
    seed: int = RANDOM_SEED


def create_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_transform(image_size: int) -> Transform:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


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
        json_path, image_dir, transform=transform, verify_images=True
    )
    if len(dataset) == 0:
        # Create empty loaders to avoid crashes downstream
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

    # Choose sensible defaults for workers/pinning
    cpu_count = max(1, (torch.get_num_threads() or 1))
    num_workers = min(4, cpu_count) - 1  # modest default; avoid oversubscription
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


# NEW: helper to mask out-of-range labels before computing loss/metrics
def _mask_valid(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return boolean mask of labels in [0, num_classes)."""
    return (labels >= 0) & (labels < num_classes)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for (img_2h, img_1h), labels in loader:
        img_2h, img_1h = img_2h.to(device), img_1h.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(img_2h, img_1h)  # safe before masking

        # Mask invalid labels to avoid "Target -2 is out of bounds."
        num_classes = outputs.size(1)
        valid_mask = _mask_valid(labels, num_classes)
        if not torch.all(valid_mask):
            bad = (~valid_mask).sum().item()
            warnings.warn(f"[train] Dropping {bad} sample(s) with invalid label(s).")
        if valid_mask.sum().item() == 0:
            continue  # skip this batch entirely

        outputs = outputs[valid_mask]
        labels = labels[valid_mask]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_n = labels.size(0)
        running_loss += float(loss.item()) * batch_n
        _, predicted = outputs.max(1)
        total += batch_n
        correct += int(predicted.eq(labels).sum().item())

    if total == 0:
        return 0.0, 0.0
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for (img_2h, img_1h), labels in loader:
            img_2h, img_1h = img_2h.to(device), img_1h.to(device)
            labels = labels.to(device).long()

            outputs = model(img_2h, img_1h)

            # Mask invalid labels at eval time as well
            num_classes = outputs.size(1)
            valid_mask = _mask_valid(labels, num_classes)
            if valid_mask.sum().item() == 0:
                continue

            outputs = outputs[valid_mask]
            labels = labels[valid_mask]

            loss = criterion(outputs, labels)
            running_loss += float(loss.item()) * labels.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += int(predicted.eq(labels).sum().item())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if total == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_best_if_exists(
    model: nn.Module, path: Path, device: torch.device
) -> nn.Module:
    if path.exists():
        return safe_load_state_dict(model, path, device)
    return model


# -----------------------------
# Orchestrator
# -----------------------------
def main(cfg: Optional[TrainConfig] = None) -> None:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = create_device()

    transform = create_transform(cfg.image_size)
    train_loader, val_loader = create_dataloaders(
        JSON_PATH, IMAGE_DIR, transform, cfg.batch_size, cfg.val_split, cfg.seed, device
    )

    model = DualResNet(num_classes=cfg.num_classes, shared_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_acc = float("-inf")
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{cfg.num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, MODEL_PATH)
            print(f"Saved new best model with Val Acc: {best_val_acc:.4f}")

    # After the loop, save the last weights if nothing was written
    if not MODEL_PATH.exists():
        save_checkpoint(model, MODEL_PATH)
        print(
            "No checkpoint was saved during training – wrote final epoch weights instead."
        )

    # Final evaluation (use best checkpoint if it fits this model)
    model = load_best_if_exists(model, MODEL_PATH, device)
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

    if val_preds.size == 0 or val_labels.size == 0:
        print(
            "No validation samples available to compute confusion matrix / classification report."
        )
        return

    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix:\n", cm)
    print(
        "Classification Report:\n",
        classification_report(val_labels, val_preds, digits=4),
    )


# ------------------------------------------------------------------
#                   Simple Qt 6 GUI (two buttons)
# ------------------------------------------------------------------
class SunsetGUI(QtWidgets.QWidget):
    """Minimal PyQt 6 front-end for training or evaluating the model."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sunset scorer")

        layout = QtWidgets.QVBoxLayout(self)
        self.img2h_path: Optional[Path] = None  # selected –2 h image
        self.img1h_path: Optional[Path] = None  # selected –1 h image
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.transform: Optional[Transform] = None

        # ── training (optional) ───────────────────────────────
        train_btn = QtWidgets.QPushButton("Train model")
        layout.addWidget(train_btn)

        # ── image pickers ─────────────────────────────────────
        picker = QtWidgets.QHBoxLayout()
        self.btn2h = QtWidgets.QPushButton("Choose –2 h image")
        self.btn1h = QtWidgets.QPushButton("Choose –1 h image")
        picker.addWidget(self.btn2h)
        picker.addWidget(self.btn1h)
        layout.addLayout(picker)

        # ── prediction ────────────────────────────────────────
        self.predict_btn = QtWidgets.QPushButton("Predict score")
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        # wiring
        train_btn.clicked.connect(self.train_model)
        self.btn2h.clicked.connect(self.choose_img2h)
        self.btn1h.clicked.connect(self.choose_img1h)
        self.predict_btn.clicked.connect(self.predict_score)

    # ---- existing functions reused --------------------------------
    def train_model(self) -> None:
        main()  # reuse existing training loop

    def load_trained_model(self) -> None:
        set_seed(RANDOM_SEED)
        device = create_device()
        transform = create_transform(IMAGE_SIZE)
        self.device = device  # stash for later
        self.transform = transform
        model = DualResNet(num_classes=NUM_CLASSES, shared_backbone=True).to(device)
        if MODEL_PATH.exists():
            safe_load_state_dict(model, MODEL_PATH, device)
        model.eval()
        self.model = model

    # ─────────── new helpers ──────────────────────────────────
    def choose_img2h(self) -> None:
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select –2 h image", str(IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img2h_path = Path(file)
            self._check_ready()

    def choose_img1h(self) -> None:
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select –1 h image", str(IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img1h_path = Path(file)
            self._check_ready()

    def _check_ready(self) -> None:
        self.predict_btn.setEnabled(bool(self.img2h_path and self.img1h_path))

    def _load_tensor(self, path: Path) -> Tensor:
        assert (
            self.transform is not None and self.device is not None
        ), "Model not initialized"
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict_score(self) -> None:
        if self.model is None:
            self.load_trained_model()
        assert (
            self.model is not None and self.device is not None
        ), "Model not initialized"
        assert (
            self.img2h_path is not None and self.img1h_path is not None
        ), "Select both images first"
        img2h = self._load_tensor(self.img2h_path)
        img1h = self._load_tensor(self.img1h_path)
        with torch.no_grad():
            logits = self.model(img2h, img1h)
            pred = int(logits.argmax(1).item()) + 1  # shift back to 1-5 for display
        QtWidgets.QMessageBox.information(
            self, "Prediction", f"Predicted sunset score: {pred}"
        )


if __name__ == "__main__":  # GUI-only entry-point
    app = QtWidgets.QApplication(sys.argv)
    gui = SunsetGUI()
    gui.show()
    sys.exit(app.exec())
