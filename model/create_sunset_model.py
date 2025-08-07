import os
import json
import random
from datetime import datetime
from typing import List, Tuple
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from pathlib import Path

# ----------------- user-configurable section -----------------

# -----------------------------
# Paths & Constants
# -----------------------------
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
JSON_PATH = parent_dir / "sunset_images_grouped.json"
IMAGE_DIR = parent_dir / "img" / "ob_hotel"
MODEL_DIR = parent_dir / "model"
MODEL_PATH = MODEL_DIR / "sunset_dual_resnet18_best.pth"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Four classes: 1,2,3, 4, 5
NUM_CLASSES = 5
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# ImageNet mean & std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def safe_load_state_dict(model: nn.Module, ckpt_path: str, device):
    """Load *compatible* weights from ``ckpt_path`` into ``model``.

    Layers whose tensor shapes don’t match (for example the final classifier
    when NUM_CLASSES has changed) are skipped instead of raising a RuntimeError.
    """
    state = torch.load(ckpt_path, map_location=device)
    own_state = model.state_dict()

    # keep only parameters whose shape matches the current model
    filtered = {k: v for k, v in state.items()
                if k in own_state and v.shape == own_state[k].shape}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(f"[safe_load_state_dict] skipped keys – "
              f"missing: {missing}  unexpected: {unexpected}")
    return model


def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SunsetDataset(Dataset):
    """PyTorch Dataset for paired sunset images (-2h and -1h) with score label."""

    def __init__(self, json_path: str, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(json_path, "r") as f:
            data = json.load(f)

        # Build samples: [(img_path_-2h, img_path_-1h, score), ...]
        self.samples: List[Tuple[str, str, int]] = []
        for entry in data:
            # Build dict by type for quick lookup
            img_dict = {img_info["type"]: img_info["name"] for img_info in entry.get("images", [])}
            if "-2h" in img_dict and "-1h" in img_dict:
                path_neg2 = os.path.join(image_dir, img_dict["-2h"])
                path_neg1 = os.path.join(image_dir, img_dict["-1h"])
                # JSON stores 1-5; shift to zero-based 0-4 for CrossEntropy
                score = int(entry["score"]) - 1
                self.samples.append((path_neg2, path_neg1, score))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        img_path_2h, img_path_1h, score = self.samples[idx]
        img_2h = self._load_image(img_path_2h)
        img_1h = self._load_image(img_path_1h)
        return (img_2h, img_1h), score


class DualResNet(nn.Module):
    """Dual-branch ResNet18 model that processes two images and outputs class logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, shared_backbone: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Output: (B, 512, 1, 1)

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

    def forward(self, img_2h, img_1h):
        f1 = self.feature_extractor(img_2h).flatten(1)
        f2 = self.feature_extractor2(img_1h).flatten(1)
        concatenated = torch.cat([f1, f2], dim=1)
        logits = self.classifier(concatenated)
        return logits


# -----------------------------
# Training & Evaluation Helpers
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for (img_2h, img_1h), labels in loader:
        img_2h, img_1h, labels = img_2h.to(device), img_1h.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(img_2h, img_1h)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (img_2h, img_1h), labels in loader:
            img_2h, img_1h, labels = img_2h.to(device), img_1h.to(device), labels.to(device)
            outputs = model(img_2h, img_1h)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ── handle empty loader to avoid ZeroDivisionError ─────────────
    if total == 0:
        return 0.0, 0.0, [], []
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# -----------------------------
# Main Script
# -----------------------------

def main():
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Dataset & Split
    dataset = SunsetDataset(JSON_PATH, IMAGE_DIR, transform=transform)
    val_size = max(1, int(len(dataset) * 0.2)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model, Loss, Optimizer
    model = DualResNet(num_classes=NUM_CLASSES, shared_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_acc = -float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best model with Val Acc: {best_val_acc:.4f}")

    # after the loop, save the last weights if nothing was written
    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)
        print("No checkpoint was saved during training – "
              "wrote final epoch weights instead.")

    # ── Final evaluation (use best checkpoint if it fits this model) ─────────
    if os.path.exists(MODEL_PATH):
        safe_load_state_dict(model, MODEL_PATH, device)
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(val_labels, val_preds, digits=4))


# ------------------------------------------------------------------
#                   Simple Qt 6 GUI (two buttons)
# ------------------------------------------------------------------

class SunsetGUI(QtWidgets.QWidget):
    """Minimal PyQt 6 front‑end for training or evaluating the model."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sunset scorer")

        layout = QtWidgets.QVBoxLayout(self)
        self.img2h_path = None          # selected –2 h image
        self.img1h_path = None          # selected –1 h image
        self.model       = None

        # ── training (optional) ───────────────────────────────
        train_btn = QtWidgets.QPushButton("Train model")
        layout.addWidget(train_btn)

        # ── image pickers ─────────────────────────────────────
        picker = QtWidgets.QHBoxLayout()
        self.btn2h = QtWidgets.QPushButton("Choose –2 h image")
        self.btn1h = QtWidgets.QPushButton("Choose –1 h image")
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
    def train_model(self):
        main()                      # reuse existing training loop

    def load_trained_model(self):
        set_seed(RANDOM_SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.device    = device              # stash for later
        self.transform = transform
        model = DualResNet(num_classes=NUM_CLASSES,
                           shared_backbone=True).to(device)
        if os.path.exists(MODEL_PATH):
            safe_load_state_dict(model, MODEL_PATH, device)
        model.eval()
        self.model = model

    # ─────────── new helpers ──────────────────────────────────
    def choose_img2h(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select –2 h image", IMAGE_DIR, "Images (*.png *.jpg *.jpeg)")
        if file:
            self.img2h_path = file
            self._check_ready()

    def choose_img1h(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select –1 h image", IMAGE_DIR, "Images (*.png *.jpg *.jpeg)")
        if file:
            self.img1h_path = file
            self._check_ready()

    def _check_ready(self):
        self.predict_btn.setEnabled(bool(self.img2h_path and self.img1h_path))

    def _load_tensor(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict_score(self):
        if self.model is None:
            self.load_trained_model()
        img2h = self._load_tensor(self.img2h_path)
        img1h = self._load_tensor(self.img1h_path)
        with torch.no_grad():
            logits = self.model(img2h, img1h)
            pred   = logits.argmax(1).item() + 1   # shift back to 1-5 for display
        QtWidgets.QMessageBox.information(
            self, "Prediction", f"Predicted sunset score: {pred}")




if __name__ == "__main__":          # GUI‑only entry‑point
    app = QtWidgets.QApplication(sys.argv)
    gui = SunsetGUI()
    gui.show()
    sys.exit(app.exec())