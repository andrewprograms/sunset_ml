# app/gui.py
"""
Main GUI for sunset_ml. Training graphs now live in a popup dialog (only shown
when 'Train model' is pressed). A separate 'Model performance…' popup shows
summary metrics for the currently loaded/best model.

Now also includes a 'Add weather data to JSON' button that uses app.weather
to backfill METAR-based weather into the dataset JSON without blocking the UI.
"""
from __future__ import annotations

# Python imports
import sys
import json
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, date

# 3rd-party imports
import torch
from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets
from torch import Tensor

BASE_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))  # allow importing top-level modules

# 1st-party imports
from app.config import RANDOM_SEED, IMAGE_SIZE, IMAGE_DIR, JSON_PATH  # type: ignore
from app.datahandler import create_transform  # type: ignore
from app.aiml import (  # type: ignore
    train_and_save,
    set_seed,
    load_model_for_inference,
    predict_class_index,
)
from app.aiml_graphs import TrainingMonitorDialog, SummaryGraphsDialog  # NEW

# Weather integration
# We import inside the worker thread so that missing 'requests' or network errors
# do not prevent the GUI from launching. (app.weather itself imports requests.)
# from app.weather import backfill_weather_for_json

# order_images helpers (optional)
try:
    from order_images import (  # type: ignore
        scan_images,
        load_existing,
        merge_into_existing,
        to_ordered_list,
        write_json,
    )
except Exception:
    scan_images = load_existing = merge_into_existing = to_ordered_list = write_json = None  # type: ignore


# ----------------------------- utilities ------------------------------------

_NEIGHBOR_WINDOW_DAYS = 14
_SUNSET_TOLERANCE_MIN = 25


def _parse_iso_local(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def _find_0h_name(images: List[Dict[str, str]]) -> Optional[str]:
    for img in images:
        if str(img.get("type")) == "0h":
            return str(img.get("name"))
    return None


def _sunset_candidate_name(entry: Dict, all_entries: List[Dict]) -> Optional[str]:
    images = entry.get("images") or []
    name0 = _find_0h_name(images)
    if name0:
        return name0
    if len(images) != 1:
        return None
    only = images[0]
    only_dt = _parse_iso_local(str(only.get("datetime", "")))
    if not only_dt:
        return None
    d0_str = str(entry.get("date", ""))
    try:
        d0 = date.fromisoformat(d0_str)
    except Exception:
        return None
    mins: List[int] = []
    for e in all_entries:
        if e is entry:
            continue
        try:
            dd = date.fromisoformat(str(e.get("date")))
        except Exception:
            continue
        if abs((dd - d0).days) > _NEIGHBOR_WINDOW_DAYS:
            continue
        n0_name = _find_0h_name(e.get("images") or [])
        if not n0_name:
            continue
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
    if not json_path.exists():
        return []
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ----------------------------- Training thread -------------------------------


class TrainingThread(QtCore.QThread):
    """
    Runs training in a background thread and emits progress events received from
    the training loop in app.aiml.train_and_save.
    """

    event = QtCore.pyqtSignal(str, dict)  # (event_name, payload)
    done = QtCore.pyqtSignal(dict)  # final summary dict

    def run(self) -> None:
        def _cb(ev: str, payload: dict) -> None:
            self.event.emit(ev, payload)

        summary = train_and_save(progress=_cb)
        self.done.emit(summary)


# ----------------------------- Review Dialog --------------------------------


class ReviewDialog(QtWidgets.QDialog):
    """Dialog to review and (optionally) edit ratings with a live preview."""

    def __init__(self, parent: QtWidgets.QWidget, image_dir: Path, json_path: Path):
        super().__init__(parent)
        self.setWindowTitle("Review sunset ratings")
        self.image_dir = image_dir
        self.json_path = json_path
        self.resize(900, 600)

        layout = QtWidgets.QHBoxLayout(self)

        # left: table + controls
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

        btns = QtWidgets.QHBoxLayout()
        self.score_buttons: List[QtWidgets.QPushButton] = []
        for i in range(1, 6):
            b = QtWidgets.QPushButton(str(i))
            b.clicked.connect(lambda _, s=i: self._set_selected_score(s))
            self.score_buttons.append(b)
            btns.addWidget(b)
        left.addLayout(btns)

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

        self.close_btn.clicked.connect(self.accept)
        self.save_btn.clicked.connect(self._save)
        self.table.itemSelectionChanged.connect(self._update_preview)
        self.search_bar.textChanged.connect(self._apply_filter)

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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
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


# ----------------------------- Weather worker/dialog -------------------------


class WeatherThread(QtCore.QThread):
    """
    Background worker that runs app.weather.backfill_weather_for_json() and
    captures its stdout so we can show a log to the user.
    """

    done = QtCore.pyqtSignal(str)  # captured log text

    def run(self) -> None:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                # Import here to avoid hard-failing GUI launch if requests isn't present
                from app.weather import backfill_weather_for_json  # type: ignore

                backfill_weather_for_json(dry_run=False, verbose=True)
        except Exception as exc:
            print(f"[weather] Error: {exc}", file=buf)
        self.done.emit(buf.getvalue())


class WeatherLogDialog(QtWidgets.QDialog):
    """Simple dialog to display the captured weather backfill log."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], text: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Weather backfill log")
        self.resize(700, 500)
        vbox = QtWidgets.QVBoxLayout(self)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlainText(text or "")
        vbox.addWidget(self.console, 1)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn)
        vbox.addLayout(row)


# ----------------------------- Rating Widget --------------------------------


class RatingWidget(QtWidgets.QWidget):
    """Step-through rater for dates that have score == -1 (unrated)."""

    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        image_dir: Path,
        json_path: Path,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.image_dir = image_dir
        self.json_path = json_path

        self.entries: List[Dict] = _load_json_safely(self.json_path)
        self.pending_indices: List[int] = [
            i
            for i, e in enumerate(self.entries)
            if int(e.get("score", -1)) == -1 and _sunset_candidate_name(e, self.entries)
        ]
        self.pos: int = 0

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

        self.back_btn.clicked.connect(self._go_back)
        self.quit_btn.clicked.connect(self._quit)
        self.review_btn.clicked.connect(self._open_review)

        self._update_view()

    def _current_index(self) -> Optional[int]:
        if 0 <= self.pos < len(self.pending_indices):
            return self.pending_indices[self.pos]
        return None

    def _update_view(self) -> None:
        idx = self._current_index()
        if idx is None:
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
        dlg = ReviewDialog(self, IMAGE_DIR, JSON_PATH)
        dlg.exec()
        # reload entries in case of edits
        self.entries = _load_json_safely(self.json_path)
        self.pending_indices = [
            i
            for i, e in enumerate(self.entries)
            if int(e.get("score", -1)) == -1 and _sunset_candidate_name(e, self.entries)
        ]
        self.pos = min(self.pos, max(0, len(self.pending_indices) - 1))
        self._update_view()


# ----------------------------- Main GUI -------------------------------------


class SunsetGUI(QtWidgets.QWidget):
    """PyQt 6 front-end for training/evaluating the model + image processing/rating."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sunset scorer")

        self.img2h_path: Optional[Path] = None
        self.img1h_path: Optional[Path] = None
        self.model: Optional[torch.nn.Module] = None
        self.device: Optional[torch.device] = None
        self.transform = create_transform(IMAGE_SIZE)

        # stacked layout: main menu <-> rater
        self.stack = QtWidgets.QStackedLayout(self)
        self.main_widget = QtWidgets.QWidget()
        self.stack.addWidget(self.main_widget)
        main_layout = QtWidgets.QVBoxLayout(self.main_widget)

        # ── top row actions ───────────────────────────────────
        top_row = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Train model")
        self.show_perf_btn = QtWidgets.QPushButton("Model performance…")  # NEW popup
        top_row.addWidget(self.train_btn)
        top_row.addWidget(self.show_perf_btn)
        top_row.addStretch(1)
        main_layout.addLayout(top_row)

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

        # ── order_images integration (if available) ───────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        main_layout.addWidget(sep)

        self.process_btn = QtWidgets.QPushButton(
            "Process new images (build/update JSON)"
        )
        if scan_images is None:
            self.process_btn.setEnabled(False)
            self.process_btn.setToolTip("order_images.py not found in project root.")
        main_layout.addWidget(self.process_btn)

        actions = QtWidgets.QHBoxLayout()
        self.rank_btn = QtWidgets.QPushButton("Rank pending dates…")
        self.review_btn = QtWidgets.QPushButton("Review all ratings…")
        actions.addWidget(self.rank_btn)
        actions.addWidget(self.review_btn)
        main_layout.addLayout(actions)

        # ── weather integration ───────────────────────────────
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        main_layout.addWidget(sep2)

        self.weather_btn = QtWidgets.QPushButton("Add weather data to JSON")
        self.weather_btn.setToolTip(
            "Fetch METAR data via AviationWeather.gov and add a 'weather' block to each date."
        )
        main_layout.addWidget(self.weather_btn)

        main_layout.addStretch(1)

        # wiring
        self.train_btn.clicked.connect(self.train_model)
        self.show_perf_btn.clicked.connect(self.show_model_performance)
        self.btn2h.clicked.connect(self.choose_img2h)
        self.btn1h.clicked.connect(self.choose_img1h)
        self.predict_btn.clicked.connect(self.predict_score)
        self.process_btn.clicked.connect(self.process_new_images)
        self.rank_btn.clicked.connect(self.open_rater)
        self.review_btn.clicked.connect(self.open_review)
        self.weather_btn.clicked.connect(self.update_weather_json)

        self.rater: Optional[RatingWidget] = None
        self._training_thread: Optional[TrainingThread] = None
        self._monitor_dialog: Optional[TrainingMonitorDialog] = None
        self._weather_thread: Optional[WeatherThread] = None

    # ---- training actions ---------------------------------------------------
    def train_model(self) -> None:
        """Show the training monitor dialog and run training in a background thread."""
        if self._training_thread is not None and self._training_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Training", "Training is already running."
            )
            return

        # Dialog is created fresh each run to show a clean timeline
        self._monitor_dialog = TrainingMonitorDialog(self)
        self._monitor_dialog.reset()

        th = TrainingThread()
        th.done.connect(self._training_finished)
        # Hook the dialog to the thread signals (live updates)
        self._monitor_dialog.connect_to(th)

        self._training_thread = th
        self._monitor_dialog.show()
        th.start()

    @QtCore.pyqtSlot(dict)
    def _training_finished(self, _: dict) -> None:
        QtWidgets.QMessageBox.information(self, "Training", "Training complete.")
        # Leave the dialog open so the user can review logs/curves; they can close manually.

    # ---- model performance summary -----------------------------------------
    def show_model_performance(self) -> None:
        dlg = SummaryGraphsDialog(self)
        dlg.exec()

    # ---- model load & predict ----------------------------------------------
    def load_trained_model(self) -> None:
        set_seed(RANDOM_SEED)
        self.model, self.device = load_model_for_inference()

    def choose_img2h(self) -> None:
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select -2 h image", str(IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img2h_path = Path(file)
            self._check_ready()

    def choose_img1h(self) -> None:
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select -1 h image", str(IMAGE_DIR), "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            self.img1h_path = Path(file)
            self._check_ready()

    def _check_ready(self) -> None:
        self.predict_btn.setEnabled(bool(self.img2h_path and self.img1h_path))

    def _load_tensor(self, path: Path) -> Tensor:
        if self.device is None:
            _, self.device = load_model_for_inference()
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict_score(self) -> None:
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
        pred = cls_idx + 1  # show 1..5
        QtWidgets.QMessageBox.information(
            self, "Prediction", f"Predicted sunset score: {pred}"
        )

    # ---- order_images integration ------------------------------------------
    def process_new_images(self) -> None:
        if scan_images is None:
            QtWidgets.QMessageBox.information(
                self, "Not available", "order_images.py not found."
            )
            return

        before_map = {e.get("date"): e for e in _load_json_safely(JSON_PATH)}
        scanned = scan_images(IMAGE_DIR)
        existing_by_date = load_existing(JSON_PATH)
        merge_into_existing(existing_by_date, scanned)
        final_list = to_ordered_list(existing_by_date)
        write_json(JSON_PATH, final_list)

        after_map = {e.get("date"): e for e in final_list}
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
        if self.rater is None:
            self.rater = RatingWidget(IMAGE_DIR, JSON_PATH, self)
            self.stack.addWidget(self.rater)
            self.rater.finished.connect(self._return_to_main)
        else:
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
        dlg = ReviewDialog(self, IMAGE_DIR, JSON_PATH)
        dlg.exec()

    # ---- weather integration -----------------------------------------------
    def update_weather_json(self) -> None:
        """Kick off weather backfill in the background and show a log when done."""
        if self._weather_thread is not None and self._weather_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Weather", "Weather update is already running."
            )
            return

        self.weather_btn.setEnabled(False)
        self.weather_btn.setText("Fetching weather…")

        th = WeatherThread()
        th.done.connect(self._weather_finished)
        self._weather_thread = th
        th.start()

    @QtCore.pyqtSlot(str)
    def _weather_finished(self, log_text: str) -> None:
        self.weather_btn.setEnabled(True)
        self.weather_btn.setText("Add weather data to JSON")

        # Show full captured log (includes per-entry messages and summary)
        dlg = WeatherLogDialog(self, log_text)
        dlg.exec()
