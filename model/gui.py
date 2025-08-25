# sunsets/model/gui.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt  # noqa: F401  (kept for potential future use)
from torch import Tensor

import config
import datahandler
import aiml


class SunsetGUI(QtWidgets.QWidget):
    """Minimal PyQt 6 front-end for training or evaluating the model."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sunset scorer")

        layout = QtWidgets.QVBoxLayout(self)
        self.img2h_path: Optional[Path] = None  # selected –2 h image
        self.img1h_path: Optional[Path] = None  # selected –1 h image
        self.model: Optional[torch.nn.Module] = None
        self.device: Optional[torch.device] = None
        self.transform = datahandler.create_transform(config.IMAGE_SIZE)

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

    # ---- actions ---------------------------------------------------
    def train_model(self) -> None:
        aiml.train_and_save()  # use current config

    def load_trained_model(self) -> None:
        aiml.set_seed(config.RANDOM_SEED)
        self.model, self.device = aiml.load_model_for_inference()

    # ─────────── helpers ───────────────────────────────────────────
    def choose_img2h(self) -> None:
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select –2 h image", str(config.IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img2h_path = Path(file)
            self._check_ready()

    def choose_img1h(self) -> None:
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select –1 h image", str(config.IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img1h_path = Path(file)
            self._check_ready()

    def _check_ready(self) -> None:
        self.predict_btn.setEnabled(bool(self.img2h_path and self.img1h_path))

    def _load_tensor(self, path: Path) -> Tensor:
        if self.device is None:
            # lazily init device to let CPU/GPU be chosen at runtime
            _, self.device = aiml.load_model_for_inference()
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict_score(self) -> None:
        if self.model is None or self.device is None:
            self.load_trained_model()
        assert self.model is not None and self.device is not None, "Model not initialized"
        assert self.img2h_path is not None and self.img1h_path is not None, "Select both images first"
        img2h = self._load_tensor(self.img2h_path)
        img1h = self._load_tensor(self.img1h_path)
        cls_idx = aiml.predict_class_index(self.model, img2h, img1h)
        pred = cls_idx + 1  # shift back to 1-5 for display
        QtWidgets.QMessageBox.information(self, "Prediction", f"Predicted sunset score: {pred}")
