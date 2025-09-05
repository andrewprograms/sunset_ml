# app/aiml_graphs.py
"""
Graphing + dialogs for training and evaluation:
- TrainingMonitorDialog: live graphs/logs while training
- SummaryGraphsDialog: on-demand popup showing how the current model performs

This file holds all matplotlib/Qt plotting to keep app/aiml.py focused on training.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import torch  # <-- use torch directly
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

# App imports
from app.config import (
    NUM_CLASSES,
    IMAGE_DIR,
    JSON_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VAL_SPLIT,
    RANDOM_SEED,
)
from app.datahandler import create_transform, create_dataloaders
from app.aiml import load_model_for_inference, evaluate, TrainConfig


# --------------------------- Training Monitor Dialog -------------------------


class TrainingMonitorDialog(QtWidgets.QDialog):
    """
    Pop-up dialog that shows a progress bar, live training curves, and a log console.
    Call .connect_to(thread) to wire it to a TrainingThread emitting (event, payload) and done(summary).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Training monitor")
        self.resize(900, 600)

        self._history_epochs: List[int] = []
        self._history_train_loss: List[float] = []
        self._history_val_loss: List[float] = []
        self._history_train_acc: List[float] = []
        self._history_val_acc: List[float] = []
        self._total_epochs: int = 0

        vbox = QtWidgets.QVBoxLayout(self)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        vbox.addWidget(self.progress)

        # Matplotlib figure
        self.fig = Figure(figsize=(6.5, 4.2), layout="constrained")
        self.ax_loss = self.fig.add_subplot(2, 1, 1)
        self.ax_acc = self.fig.add_subplot(2, 1, 2)

        self.ax_loss.set_ylabel("Loss")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_xlabel("Epoch")

        (self.line_tl,) = self.ax_loss.plot([], [], label="Train Loss")
        (self.line_vl,) = self.ax_loss.plot([], [], label="Val Loss")
        self.ax_loss.legend(loc="upper right")

        (self.line_ta,) = self.ax_acc.plot([], [], label="Train Acc")
        (self.line_va,) = self.ax_acc.plot([], [], label="Val Acc")
        self.ax_acc.legend(loc="lower right")

        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(NavigationToolbar2QT(self.canvas, self))
        vbox.addWidget(self.canvas, 1)

        # Log console
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Training logs will appear here…")
        self.console.setMinimumHeight(120)
        vbox.addWidget(self.console)

        # Buttons
        btnrow = QtWidgets.QHBoxLayout()
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btnrow.addStretch(1)
        btnrow.addWidget(self.close_btn)
        vbox.addLayout(btnrow)

    # ---- public API ---------------------------------------------------------

    def reset(self) -> None:
        self.progress.setValue(0)
        self.console.clear()
        self._history_epochs.clear()
        self._history_train_loss.clear()
        self._history_val_loss.clear()
        self._history_train_acc.clear()
        self._history_val_acc.clear()
        self._total_epochs = 0
        self._redraw()

    def connect_to(self, thread: QtCore.QThread) -> None:
        """
        Wire this dialog to a TrainingThread that exposes:
          - event: pyqtSignal(str, dict)
          - done:  pyqtSignal(dict)
        """
        event_sig = getattr(thread, "event", None)
        done_sig = getattr(thread, "done", None)
        if event_sig is not None:
            event_sig.connect(self.on_event)  # type: ignore[arg-type]
        if done_sig is not None:
            done_sig.connect(self.on_done)  # type: ignore[arg-type]

    # ---- slots to handle training events -----------------------------------

    @QtCore.pyqtSlot(str, dict)
    def on_event(self, event: str, payload: dict) -> None:
        if event == "log":
            txt = payload.get("text", "")
            if txt:
                self.console.appendPlainText(str(txt))
        elif event == "epoch":
            ep = int(payload.get("epoch", 0))
            tot = max(1, int(payload.get("total_epochs", 1)))
            self._total_epochs = tot

            tl = float(payload.get("train_loss", 0.0))
            vl = float(payload.get("val_loss", 0.0))
            ta = float(payload.get("train_acc", 0.0))
            va = float(payload.get("val_acc", 0.0))

            self._history_epochs.append(ep)
            self._history_train_loss.append(tl)
            self._history_val_loss.append(vl)
            self._history_train_acc.append(ta)
            self._history_val_acc.append(va)

            self.progress.setValue(int(100 * ep / tot))
            self._redraw()
        elif event == "checkpoint":
            va = payload.get("val_acc")
            if va is not None:
                self.console.appendPlainText(f"[checkpoint] New best Val Acc: {va:.4f}")
        elif event == "final":
            report = payload.get("classification_report", "")
            cm = payload.get("confusion_matrix")
            if report:
                self.console.appendPlainText(
                    "\n=== Classification Report ===\n" + str(report)
                )
            if cm is not None:
                self.console.appendPlainText("=== Confusion Matrix ===")
                self.console.appendPlainText(str(cm))

    @QtCore.pyqtSlot(dict)
    def on_done(self, _: dict) -> None:
        self.console.appendPlainText("\nTraining complete.")

    # ---- internals ----------------------------------------------------------

    def _redraw(self) -> None:
        self.line_tl.set_data(self._history_epochs, self._history_train_loss)
        self.line_vl.set_data(self._history_epochs, self._history_val_loss)
        self.line_ta.set_data(self._history_epochs, self._history_train_acc)
        self.line_va.set_data(self._history_epochs, self._history_val_acc)

        if self._history_epochs:
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            self.ax_acc.relim()
            self.ax_acc.autoscale_view()

        self.canvas.draw_idle()


# --------------------------- Summary Graphs Dialog ---------------------------


class SummaryGraphsDialog(QtWidgets.QDialog):
    """
    Popup that evaluates the CURRENT best model on the validation split and shows:
      - Confusion matrix (heatmap)
      - Per-class F1 bar chart
      - Overall accuracy
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model performance summary")
        self.resize(950, 640)

        vbox = QtWidgets.QVBoxLayout(self)

        # status label (shows overall accuracy and dataset info)
        self.status_label = QtWidgets.QLabel("")
        vbox.addWidget(self.status_label)

        # two figures: cm + per-class F1
        self.fig_cm = Figure(figsize=(5.8, 4.5), layout="constrained")
        self.ax_cm = self.fig_cm.add_subplot(1, 1, 1)
        self.canvas_cm = FigureCanvas(self.fig_cm)

        self.fig_bar = Figure(figsize=(5.8, 4.5), layout="constrained")
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.canvas_bar = FigureCanvas(self.fig_bar)

        row = QtWidgets.QHBoxLayout()
        col1 = QtWidgets.QVBoxLayout()
        col2 = QtWidgets.QVBoxLayout()
        col1.addWidget(NavigationToolbar2QT(self.canvas_cm, self))
        col1.addWidget(self.canvas_cm, 1)
        col2.addWidget(NavigationToolbar2QT(self.canvas_bar, self))
        col2.addWidget(self.canvas_bar, 1)
        row.addLayout(col1, 1)
        row.addLayout(col2, 1)
        vbox.addLayout(row, 1)

        # buttons
        btnrow = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.close_btn = QtWidgets.QPushButton("Close")
        self.refresh_btn.clicked.connect(self.refresh)
        self.close_btn.clicked.connect(self.accept)
        btnrow.addStretch(1)
        btnrow.addWidget(self.refresh_btn)
        btnrow.addWidget(self.close_btn)
        vbox.addLayout(btnrow)

        # first load
        QtCore.QTimer.singleShot(0, self.refresh)

    # ---- logic --------------------------------------------------------------

    def refresh(self) -> None:
        """
        Load best model, evaluate on the current validation split, draw charts.
        """
        try:
            # 1) Get model & device first (so we can build loaders with the correct device)
            model, device = load_model_for_inference()

            # 2) Build loaders using that device (previously passed None → caused 'NoneType'.type)
            cfg = TrainConfig(
                image_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                val_split=VAL_SPLIT,
                seed=RANDOM_SEED,
            )
            transform = create_transform(cfg.image_size)
            _, val_loader = create_dataloaders(
                JSON_PATH,
                IMAGE_DIR,
                transform,
                cfg.batch_size,
                cfg.val_split,
                cfg.seed,
                device,
            )

            # 3) Evaluate
            criterion = torch.nn.CrossEntropyLoss()
            val_loss, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )

            if val_preds.size == 0 or val_labels.size == 0:
                self._render_empty("No validation samples available.")
                return

            # Metrics
            from sklearn.metrics import classification_report, confusion_matrix

            cm = confusion_matrix(
                val_labels, val_preds, labels=list(range(NUM_CLASSES))
            )
            report_dict = classification_report(
                val_labels,
                val_preds,
                labels=list(range(NUM_CLASSES)),
                output_dict=True,
                zero_division=0,
            )

            overall_acc = float(report_dict.get("accuracy", float(val_acc)))

            self._draw_confusion_matrix(cm)
            self._draw_f1_bars(report_dict)

            self.status_label.setText(
                f"Validation loss: {val_loss:.4f}    |    Accuracy: {overall_acc:.4f}    "
                f"|    Classes: {NUM_CLASSES}    |    Samples: {int(np.array(val_labels).size)}"
            )
        except Exception as exc:
            self._render_empty(f"Error while computing summary: {exc}")

    def _draw_confusion_matrix(self, cm: np.ndarray) -> None:
        self.ax_cm.clear()
        im = self.ax_cm.imshow(cm, interpolation="nearest")
        self.ax_cm.set_title("Confusion matrix")
        self.fig_cm.colorbar(im, ax=self.ax_cm, fraction=0.046, pad=0.04)
        ticks = np.arange(NUM_CLASSES)
        self.ax_cm.set_xticks(ticks)
        self.ax_cm.set_yticks(ticks)
        self.ax_cm.set_xticklabels([str(i + 1) for i in ticks])
        self.ax_cm.set_yticklabels([str(i + 1) for i in ticks])
        self.ax_cm.set_xlabel("Predicted")
        self.ax_cm.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax_cm.text(j, i, int(cm[i, j]), ha="center", va="center")
        self.canvas_cm.draw_idle()

    def _draw_f1_bars(self, report_dict: Dict) -> None:
        self.ax_bar.clear()
        f1s: List[float] = []
        labels: List[str] = []
        for idx in range(NUM_CLASSES):
            key = str(idx)
            per = report_dict.get(key)
            if isinstance(per, dict):
                labels.append(str(idx + 1))  # display classes as 1..N
                f1s.append(float(per.get("f1-score", 0.0)))
        x = np.arange(len(f1s))
        self.ax_bar.bar(x, f1s)
        self.ax_bar.set_title("Per-class F1 score")
        self.ax_bar.set_xlabel("Class")
        self.ax_bar.set_ylabel("F1")
        self.ax_bar.set_xticks(x, labels)
        self.ax_bar.set_ylim(0, 1.0)
        self.canvas_bar.draw_idle()

    def _render_empty(self, message: str) -> None:
        self.ax_cm.clear()
        self.ax_cm.set_title("Confusion matrix")
        self.ax_cm.text(
            0.5, 0.5, message, ha="center", va="center", transform=self.ax_cm.transAxes
        )
        self.ax_cm.axis("off")
        self.canvas_cm.draw_idle()

        self.ax_bar.clear()
        self.ax_bar.set_title("Per-class F1 score")
        self.ax_bar.text(
            0.5, 0.5, message, ha="center", va="center", transform=self.ax_bar.transAxes
        )
        self.ax_bar.axis("off")
        self.canvas_bar.draw_idle()
