# sunset_ml/app/aiml.py
"""
Utilities for training and evaluating sunset_ml model
"""
from __future__ import annotations

# Python Imports
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# 3rd Party Imports
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor
from torchvision import models

# 1st Party Imports
BASE_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
# print("REPO_DIR:", BASE_DIR)
from app.config import (
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    VAL_SPLIT,
    RANDOM_SEED,
    JSON_PATH,
    IMAGE_DIR,
    MODEL_PATH,
)
from app.datahandler import create_transform, create_dataloaders


# -----------------------------
# Seed / Device / Checkpoints
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """Set seed for app"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_device() -> torch.device:
    """Create torch device for app (uses GPU if available)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def safe_load_state_dict(
    model: nn.Module, ckpt_path: Path, device: torch.device
) -> nn.Module:
    """Load only compatible weights from checkpoint."""
    state = torch.load(ckpt_path, map_location=device)
    own_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in own_state and v.shape == own_state[k].shape
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(
            f"[safe_load_state_dict] skipped keys - missing: {missing}  unexpected: {unexpected}"
        )
    return model


def load_best_if_exists(
    model: nn.Module, path: Path, device: torch.device
) -> nn.Module:
    """Load model checkpoint from disk if it exists."""
    if path.exists():
        return safe_load_state_dict(model, path, device)
    return model


# -----------------------------
# Model
# -----------------------------
class DualResNet(nn.Module):
    """Dual-branch ResNet18 model that processes two images and outputs class logits."""

    def __init__(
        self, num_classes: int = NUM_CLASSES, shared_backbone: bool = True
    ) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )  # (B,512,1,1)
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
        """Forward pass."""
        f1 = self.feature_extractor(img_2h).flatten(1)
        f2 = self.feature_extractor2(img_1h).flatten(1)
        concatenated = torch.cat([f1, f2], dim=1)
        logits = self.classifier(concatenated)
        return logits


# -----------------------------
# Training / Evaluation
# -----------------------------
@dataclass(frozen=True)
class TrainConfig:
    """Training config."""
    image_size: int = IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    val_split: float = VAL_SPLIT
    num_classes: int = NUM_CLASSES
    seed: int = RANDOM_SEED


def _mask_valid(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Mask out invalid labels."""
    return (labels >= 0) & (labels < num_classes)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for (img_2h, img_1h), labels in loader:
        img_2h, img_1h = img_2h.to(device), img_1h.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(img_2h, img_1h)

        valid_mask = _mask_valid(labels, outputs.size(1))
        if not torch.all(valid_mask):
            warnings.warn(
                f"[train] Dropping {(~valid_mask).sum().item()} invalid label(s)."
            )
        if valid_mask.sum().item() == 0:
            continue

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
    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Function to evaluate sunset_ml model."""
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

            valid_mask = _mask_valid(labels, outputs.size(1))
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
    return (
        running_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def train_and_save(cfg: Optional[TrainConfig] = None) -> None:
    """End-to-end training loop that saves the best checkpoint and prints a report."""
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

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, MODEL_PATH)
            print(f"Saved new best model with Val Acc: {best_val_acc:.4f}")

    if not MODEL_PATH.exists():
        save_checkpoint(model, MODEL_PATH)
        print(
            "No checkpoint was saved during training - wrote final epoch weights instead."
        )

    # Final evaluation using best checkpoint (when compatible)
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


# -----------------------------
# Inference helpers
# -----------------------------
def load_model_for_inference(
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.device]:
    """Load trained model for inference."""
    device = device or create_device()
    model = DualResNet(num_classes=NUM_CLASSES, shared_backbone=True).to(device)
    if MODEL_PATH.exists():
        safe_load_state_dict(model, MODEL_PATH, device)
    model.eval()
    return model, device


def predict_class_index(model: nn.Module, img2h: Tensor, img1h: Tensor) -> int:
    """Predict class index for a pair of images."""
    with torch.no_grad():
        logits = model(img2h, img1h)
        return int(logits.argmax(1).item())  # 0..4
