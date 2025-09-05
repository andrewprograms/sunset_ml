# sunsets/model/gui.py
"""
GUI front-end for sunset_ml app, extended with image processing and rating workflow.

- "Process new images" runs the same logic as order_images.py to scan the img folder
  and update the grouped JSON.
- If new dates/images were added, you'll be asked to rate them now.
- The rating flow shows the day's sunset image (explicit 0h, or inferred for single-image days)
  and lets you assign a 1–5 score, with Back/Review/Quit controls,
  then returns to the main menu when done.

This file assumes the repository layout used in the original sources where this module
resides under ".../model/gui.py" and the project's base directory (two parents up)
contains `order_images.py`, `img/`, and `sunset_images_grouped.json`.
"""
from __future__ import annotations

# Python imports
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta

# 3rd-party imports
import torch
from PIL import Image, UnidentifiedImageError
from PyQt6 import QtCore, QtGui, QtWidgets
from torch import Tensor

BASE_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.append(
    str(BASE_DIR)
)  # allow importing top-level modules (e.g., order_images.py)

# 1st-party imports
from app.config import RANDOM_SEED, IMAGE_SIZE, IMAGE_DIR, JSON_PATH  # type: ignore
from app.datahandler import create_transform  # type: ignore
from app.aiml import (  # type: ignore
    train_and_save,
    set_seed,
    load_model_for_inference,
    predict_class_index,
)

# order_images helpers (we'll call the pure functions directly)
from order_images import (  # type: ignore
    scan_images,
    load_existing,
    merge_into_existing,
    to_ordered_list,
    write_json,
)


# ----------------------------- utilities ------------------------------------

# Heuristics for inferring sunset when a day has only a single image.
_NEIGHBOR_WINDOW_DAYS = 14
_SUNSET_TOLERANCE_MIN = 25  # minutes difference allowed vs neighbor 0h median


def _parse_iso_local(dt_str: str) -> Optional[datetime]:
    try:
        # ImageRecord.datetime is ISO 8601 string; treat as naive local time
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def _find_0h_name(images: List[Dict[str, str]]) -> Optional[str]:
    """Return the filename for the '0h' image in a day's image list."""
    for img in images:
        if str(img.get("type")) == "0h":
            return str(img.get("name"))
    return None


def _sunset_candidate_name(entry: Dict, all_entries: List[Dict]) -> Optional[str]:
    """
    Return the name of the sunset image to display for rating:
    - Prefer the explicit '0h' image if present.
    - If the day has exactly one image, infer sunset using neighbor days' 0h times:
      compute the median 0h time-of-day from entries within ±NEIGHBOR_WINDOW_DAYS
      and accept if within ±SUNSET_TOLERANCE_MIN of the single image's time-of-day.
    Otherwise return None (not rankable).
    """
    images = entry.get("images") or []
    # Explicit 0h takes precedence
    name0 = _find_0h_name(images)
    if name0:
        return name0
    # Infer only if it's a single-image day
    if len(images) != 1:
        return None
    only = images[0]
    only_dt = _parse_iso_local(str(only.get("datetime", "")))
    if not only_dt:
        return None
    # Collect neighbor 0h times within window
    d0_str = str(entry.get("date", ""))
    try:
        d0 = date.fromisoformat(d0_str)
    except Exception:
        return None
    mins: List[int] = []
    for e in all_entries:
        if e is entry:
            continue
        ds = e.get("date")
        try:
            dd = date.fromisoformat(str(ds))
        except Exception:
            continue
        if abs((dd - d0).days) > _NEIGHBOR_WINDOW_DAYS:
            continue
        # get this neighbor's 0h
        n0_name = _find_0h_name(e.get("images") or [])
        if not n0_name:
            continue
        # find datetime for that 0h
        for im in e.get("images") or []:
            if str(im.get("name")) == n0_name:
                ndt = _parse_iso_local(str(im.get("datetime", "")))
                if ndt:
                    mins.append(ndt.hour * 60 + ndt.minute)
                break
    if not mins:
        return None
    mins.sort()
    median_min = mins[len(mins) // 2]
    only_min = only_dt.hour * 60 + only_dt.minute
    if abs(only_min - median_min) <= _SUNSET_TOLERANCE_MIN:
        return str(only.get("name"))
    return None


def _load_json_safely(json_path: Path) -> List[Dict]:
    """Load JSON list or return empty list."""
    if not json_path.exists():
        return []
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ----------------------------- Review Dialog --------------------------------


class ReviewDialog(QtWidgets.QDialog):
    """Dialog to review and (optionally) edit ratings with a live preview."""

    def __init__(self, parent: QtWidgets.QWidget, image_dir: Path, json_path: Path):
        super().__init__(parent)
        self.setWindowTitle("Review sunset ratings")
        self.image_dir = image_dir
        self.json_path = json_path
        self.resize(900, 600)

        # layout
        layout = QtWidgets.QHBoxLayout(self)

        # left: table
        left = QtWidgets.QVBoxLayout()
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Filter by date (YYYY-MM-DD) or score...")
        left.addWidget(self.search_bar)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Date", "Score", "# Images"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        left.addWidget(self.table, 1)

        # quick-set buttons
        btns = QtWidgets.QHBoxLayout()
        self.score_buttons: List[QtWidgets.QPushButton] = []
        for i in range(1, 6):
            b = QtWidgets.QPushButton(str(i))
            b.clicked.connect(lambda _, s=i: self._set_selected_score(s))
            self.score_buttons.append(b)
            btns.addWidget(b)
        left.addLayout(btns)

        # save/close
        action_bar = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save")
        self.close_btn = QtWidgets.QPushButton("Close")
        action_bar.addStretch(1)
        action_bar.addWidget(self.save_btn)
        action_bar.addWidget(self.close_btn)
        left.addLayout(action_bar)

        layout.addLayout(left, 3)

        # right: preview
        right = QtWidgets.QVBoxLayout()
        self.preview_label = QtWidgets.QLabel("Preview")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(360, 240)
        self.preview_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        right.addWidget(self.preview_label, 1)
        layout.addLayout(right, 2)

        # wiring
        self.close_btn.clicked.connect(self.accept)
        self.save_btn.clicked.connect(self._save)
        self.table.itemSelectionChanged.connect(self._update_preview)
        self.search_bar.textChanged.connect(self._apply_filter)

        # data
        self._all_entries: List[Dict] = _load_json_safely(self.json_path)
        self._filtered_indices: List[int] = list(range(len(self._all_entries)))
        self._populate_table()

    # ----- internals -----
    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for idx in self._filtered_indices:
            entry = self._all_entries[idx]
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(str(entry.get("date", "")))
            )
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(entry.get("score", -1)))
            )
            images = entry.get("images") or []
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(len(images))))
        if self.table.rowCount():
            self.table.selectRow(0)
        self._update_preview()

    def _apply_filter(self, text: str) -> None:
        t = (text or "").strip().lower()
        self._filtered_indices = []
        for i, e in enumerate(self._all_entries):
            date_ok = t in str(e.get("date", "")).lower()
            score_ok = t in str(e.get("score", -1)).lower()
            if not t or date_ok or score_ok:
                self._filtered_indices.append(i)
        self._populate_table()

    def _selected_entry_index(self) -> Optional[int]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self._filtered_indices):
            return self._filtered_indices[row]
        return None

    def _update_preview(self) -> None:
        idx = self._selected_entry_index()
        if idx is None:
            self.preview_label.setText("Preview")
            return
        entry = self._all_entries[idx]
        # Prefer explicit 0h; otherwise show inferred single-image sunset if available
        name = _sunset_candidate_name(entry, self._all_entries)
        if not name:
            self.preview_label.setText("No sunset image available")
            return
        path = self.image_dir / name
        if not path.exists():
            self.preview_label.setText(f"Image not found:\n{path.name}")
            return
        pix = QtGui.QPixmap(str(path))
        if pix.isNull():
            self.preview_label.setText("Could not load image")
            return
        self.preview_label.setPixmap(
            pix.scaled(
                self.preview_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # keep preview scaled
        super().resizeEvent(event)
        self._update_preview()

    def _set_selected_score(self, score: int) -> None:
        idx = self._selected_entry_index()
        if idx is None:
            return
        self._all_entries[idx]["score"] = int(score)
        self._populate_table()

    def _save(self) -> None:
        try:
            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(self._all_entries, f, indent=4, ensure_ascii=False)
            QtWidgets.QMessageBox.information(self, "Saved", "Ratings saved.")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save JSON:\n{exc}"
            )


# ----------------------------- Rating Widget --------------------------------


class RatingWidget(QtWidgets.QWidget):
    """Step-through rater for dates that have score == -1 (unrated)."""

    finished = QtCore.pyqtSignal()  # emitted when user is done or quits

    def __init__(
        self,
        image_dir: Path,
        json_path: Path,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.image_dir = image_dir
        self.json_path = json_path

        # state
        self.entries: List[Dict] = _load_json_safely(self.json_path)
        self.pending_indices: List[int] = [
            i
            for i, e in enumerate(self.entries)
            if int(e.get("score", -1)) == -1 and _sunset_candidate_name(e, self.entries)
        ]
        self.pos: int = 0

        # ui
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.back_btn = QtWidgets.QPushButton("Back")
        self.review_btn = QtWidgets.QPushButton("Review…")
        self.quit_btn = QtWidgets.QPushButton("Quit")
        header.addWidget(self.back_btn)
        header.addStretch(1)
        header.addWidget(self.review_btn)
        header.addWidget(self.quit_btn)
        layout.addLayout(header)

        self.day_label = QtWidgets.QLabel("")
        self.day_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.day_label.font()
        font.setPointSize(font.pointSize() + 3)
        font.setBold(True)
        self.day_label.setFont(font)
        layout.addWidget(self.day_label)

        self.image_label = QtWidgets.QLabel("")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 320)
        self.image_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout.addWidget(self.image_label, 1)

        # rating buttons
        rate_row = QtWidgets.QHBoxLayout()
        self.rate_buttons: List[QtWidgets.QPushButton] = []
        for i in range(1, 6):
            b = QtWidgets.QPushButton(str(i))
            b.setFixedHeight(48)
            b.setDefault(i == 3)
            b.clicked.connect(lambda _, s=i: self._rate_and_next(s))
            self.rate_buttons.append(b)
            rate_row.addWidget(b)
        layout.addLayout(rate_row)

        # keyboard shortcuts 1..5
        for i in range(1, 6):
            sh = QtGui.QShortcut(QtGui.QKeySequence(str(i)), self)
            sh.activated.connect(lambda s=i: self._rate_and_next(s))

        # wiring
        self.back_btn.clicked.connect(self._go_back)
        self.quit_btn.clicked.connect(self._quit)
        self.review_btn.clicked.connect(self._open_review)

        self._update_view()

    # ----- behavior -----
    def _current_index(self) -> Optional[int]:
        if 0 <= self.pos < len(self.pending_indices):
            return self.pending_indices[self.pos]
        return None

    def _update_view(self) -> None:
        idx = self._current_index()
        if idx is None:
            # nothing pending
            self.day_label.setText("All caught up!")
            self.image_label.setText("No unrated dates remain.")
            for b in self.rate_buttons:
                b.setEnabled(False)
            self.back_btn.setEnabled(False)
            return
        entry = self.entries[idx]
        self.day_label.setText(
            f"Date: {entry.get('date', '')}    (item {self.pos+1}/{len(self.pending_indices)})"
        )
        name = _sunset_candidate_name(entry, self.entries)
        if not name:
            self.image_label.setText("No sunset image available")
        else:
            path = self.image_dir / name
            pix = QtGui.QPixmap(str(path))
            if pix.isNull():
                self.image_label.setText(f"Could not load:\n{path.name}")
            else:
                self.image_label.setPixmap(
                    pix.scaled(
                        self.image_label.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                )
        self.back_btn.setEnabled(self.pos > 0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # re-scale current pixmap to fit
        if not self.image_label.pixmap():
            return
        self._update_view()

    def _save(self) -> None:
        try:
            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=4, ensure_ascii=False)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Could not save JSON:\n{exc}"
            )

    def _rate_and_next(self, score: int) -> None:
        idx = self._current_index()
        if idx is None:
            self.finished.emit()
            return
        self.entries[idx]["score"] = int(score)
        self._save()
        # remove from pending and do not advance pos unnecessarily
        self.pending_indices.pop(self.pos)
        if self.pos >= len(self.pending_indices) and self.pos > 0:
            self.pos -= 1
        self._update_view()
        if not self.pending_indices:
            QtWidgets.QMessageBox.information(
                self, "Done", "Finished rating all new images."
            )
            self.finished.emit()

    def _go_back(self) -> None:
        if self.pos > 0:
            self.pos -= 1
            self._update_view()

    def _quit(self) -> None:
        self.finished.emit()

    def _open_review(self) -> None:
        dlg = ReviewDialog(self, self.image_dir, self.json_path)
        dlg.exec()
        # reload entries in case of edits
        self.entries = _load_json_safely(self.json_path)
        # Recompute pending list (any rows with -1 score and a sunset candidate)
        self.pending_indices = [
            i
            for i, e in enumerate(self.entries)
            if int(e.get("score", -1)) == -1 and _sunset_candidate_name(e, self.entries)
        ]
        # Clamp pos
        self.pos = min(self.pos, max(0, len(self.pending_indices) - 1))
        self._update_view()


# ----------------------------- Main GUI -------------------------------------


class SunsetGUI(QtWidgets.QWidget):
    """PyQt 6 front-end for training/evaluating the model + image processing/rating."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sunset scorer")

        self.img2h_path: Optional[Path] = None  # selected -2 h image
        self.img1h_path: Optional[Path] = None  # selected -1 h image
        self.model: Optional[torch.nn.Module] = None
        self.device: Optional[torch.device] = None
        self.transform = create_transform(IMAGE_SIZE)

        # stacked layout: main menu <-> rater
        self.stack = QtWidgets.QStackedLayout(self)

        self.main_widget = QtWidgets.QWidget()
        self.stack.addWidget(self.main_widget)

        main_layout = QtWidgets.QVBoxLayout(self.main_widget)

        # ── training (optional) ───────────────────────────────
        train_btn = QtWidgets.QPushButton("Train model")
        main_layout.addWidget(train_btn)

        # ── image pickers ─────────────────────────────────────
        picker = QtWidgets.QHBoxLayout()
        self.btn2h = QtWidgets.QPushButton("Choose -2 h image")
        self.btn1h = QtWidgets.QPushButton("Choose -1 h image")
        picker.addWidget(self.btn2h)
        picker.addWidget(self.btn1h)
        main_layout.addLayout(picker)

        # ── prediction ────────────────────────────────────────
        self.predict_btn = QtWidgets.QPushButton("Predict score")
        self.predict_btn.setEnabled(False)
        main_layout.addWidget(self.predict_btn)

        # ── new: order_images integration ─────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        main_layout.addWidget(sep)

        self.process_btn = QtWidgets.QPushButton(
            "Process new images (build/update JSON)"
        )
        main_layout.addWidget(self.process_btn)

        actions = QtWidgets.QHBoxLayout()
        self.rank_btn = QtWidgets.QPushButton("Rank pending dates…")
        self.review_btn = QtWidgets.QPushButton("Review all ratings…")
        actions.addWidget(self.rank_btn)
        actions.addWidget(self.review_btn)
        main_layout.addLayout(actions)
        main_layout.addStretch(1)

        # wiring
        train_btn.clicked.connect(self.train_model)
        self.btn2h.clicked.connect(self.choose_img2h)
        self.btn1h.clicked.connect(self.choose_img1h)
        self.predict_btn.clicked.connect(self.predict_score)

        self.process_btn.clicked.connect(self.process_new_images)
        self.rank_btn.clicked.connect(self.open_rater)
        self.review_btn.clicked.connect(self.open_review)

        # rater placeholder (added lazily when needed)
        self.rater: Optional[RatingWidget] = None

    # ---- actions ---------------------------------------------------
    def train_model(self) -> None:
        """Train and save model."""
        train_and_save()  # use current config

    def load_trained_model(self) -> None:
        """Load trained model."""
        set_seed(RANDOM_SEED)
        self.model, self.device = load_model_for_inference()

    # ─────────── helpers ───────────────────────────────────────────
    def choose_img2h(self) -> None:
        """Select -2 h image."""
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select -2 h image", str(IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img2h_path = Path(file)
            self._check_ready()

    def choose_img1h(self) -> None:
        """Select -1 h image."""
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select -1 h image", str(IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img1h_path = Path(file)
            self._check_ready()

    def _check_ready(self) -> None:
        """Check if both images selected, and enable predict button."""
        self.predict_btn.setEnabled(bool(self.img2h_path and self.img1h_path))

    def _load_tensor(self, path: Path) -> Tensor:
        """Load image as tensor."""
        if self.device is None:
            # lazily init device to let CPU/GPU be chosen at runtime
            _, self.device = load_model_for_inference()
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict_score(self) -> None:
        """Predict score for a pair of sunset images."""
        if self.model is None or self.device is None:
            self.load_trained_model()
        assert (
            self.model is not None and self.device is not None
        ), "Model not initialized"
        assert (
            self.img2h_path is not None and self.img1h_path is not None
        ), "Select both images first"
        img2h = self._load_tensor(self.img2h_path)
        img1h = self._load_tensor(self.img1h_path)
        cls_idx = predict_class_index(self.model, img2h, img1h)
        pred = cls_idx + 1  # shift back to 1-5 for display
        QtWidgets.QMessageBox.information(
            self, "Prediction", f"Predicted sunset score: {pred}"
        )

    # ─────────── order_images integration ──────────────────────────
    def process_new_images(self) -> None:
        """
        Scan the image directory and merge into JSON (same logic as order_images.py).
        If changes were made, ask to start rating.
        """
        # Before
        before_map = {e.get("date"): e for e in _load_json_safely(JSON_PATH)}

        # Compute new scan + merge using the shared helpers
        scanned = scan_images(IMAGE_DIR)
        existing_by_date = load_existing(JSON_PATH)
        merge_into_existing(existing_by_date, scanned)
        final_list = to_ordered_list(existing_by_date)
        write_json(JSON_PATH, final_list)

        # After
        after_map = {e.get("date"): e for e in final_list}
        # Changes metrics
        new_dates = [d for d in after_map.keys() if d not in before_map]
        new_images = 0
        for d, entry in after_map.items():
            after_names = {img["name"] for img in entry.get("images", [])}
            before_names = {
                img["name"] for img in (before_map.get(d, {}).get("images", []) or [])
            }
            new_images += max(0, len(after_names - before_names))

        msg = f"Processed images.\nNew dates: {len(new_dates)}\nNew images: {new_images}\nJSON: {JSON_PATH}"
        if new_dates or new_images:
            ret = QtWidgets.QMessageBox.question(
                self,
                "Images processed",
                msg + "\n\nWould you like to rank new dates now?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            if ret == QtWidgets.QMessageBox.StandardButton.Yes:
                self.open_rater()
        else:
            QtWidgets.QMessageBox.information(self, "No changes", msg)

    def open_rater(self) -> None:
        """Open rating widget for entries with score == -1 and an available sunset image."""
        # Construct or reuse rater
        if self.rater is None:
            self.rater = RatingWidget(IMAGE_DIR, JSON_PATH, self)
            self.stack.addWidget(self.rater)
            self.rater.finished.connect(self._return_to_main)
        else:
            # refresh its state
            self.stack.setCurrentWidget(self.rater)
            self.rater.entries = _load_json_safely(JSON_PATH)
            self.rater.pending_indices = [
                i
                for i, e in enumerate(self.rater.entries)
                if int(e.get("score", -1)) == -1
                and _sunset_candidate_name(e, self.rater.entries)
            ]
            self.rater.pos = 0
            self.rater._update_view()

        self.stack.setCurrentWidget(self.rater)

    def _return_to_main(self) -> None:
        self.stack.setCurrentWidget(self.main_widget)

    def open_review(self) -> None:
        """Open the review dialog (all entries)."""
        dlg = ReviewDialog(self, IMAGE_DIR, JSON_PATH)
        dlg.exec()
