#!/usr/bin/env python3
"""
Order/merge sunset images by date and write a grouped JSON manifest.

- No CLI: just run the file (e.g., in VS Code).
- Base folder is the folder this script lives in (expected to be ".../sunsets").
- Preserves any existing scores in the output JSON.
- Adds new dates with score = -1 and ensures every entry has a score.
- Classifies each day's images so the most recent is "0h",
  the previous is "-1h", the one before is "-2h", and all earlier
  ones are also "-2h" (same logic as the original script).

Filename format expected:
    YY_MM_DD--HH_MM_SS.jpg

Example:
    24_08_12--19_21_07.jpg
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, TypedDict
import json
import logging
import sys


# ----------------- user-configurable defaults (relative to this script) -----
IMG_SUBDIR = Path("img/ob_hotel")  # where the JPGs are
JSON_NAME = "sunset_images_grouped.json"  # output file name
# ---------------------------------------------------------------------------


# ------------------------------- typing -------------------------------------


class ImageRecord(TypedDict):
    name: str
    datetime: str  # ISO 8601 string
    type: str  # "0h", "-1h", "-2h"


class DateEntry(TypedDict):
    date: str  # ISO date, e.g. "2024-08-12"
    score: int
    images: List[ImageRecord]


@dataclass(frozen=True)
class ScannedImage:
    name: str
    dt: datetime


# ---------------------------- helpers & logic -------------------------------


def script_base_dir() -> Path:
    """
    Base directory for the project, i.e., the folder containing this script.
    Expected to be the 'sunsets' folder.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is unavailable (rare when run as a proper script)
        return Path.cwd()


def parse_filename_to_datetime(filename: str) -> Optional[datetime]:
    """
    Parse a filename of the form 'YY_MM_DD--HH_MM_SS.jpg' into a datetime.
    Returns None if parsing fails.
    """
    stem = Path(filename).stem  # drops extension
    if "--" not in stem:
        return None
    date_str, time_str = stem.split("--", 1)
    try:
        return datetime.strptime(date_str + time_str, "%y_%m_%d%H_%M_%S")
    except ValueError:
        return None


def scan_images(img_dir: Path) -> Dict[date, List[ScannedImage]]:
    """
    Walk the image directory and group images by calendar day.
    Only '.jpg' files (case-insensitive) are considered, matching original behavior.
    """
    images_by_date: Dict[date, List[ScannedImage]] = {}

    for path in img_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".jpg":
            continue

        dt = parse_filename_to_datetime(path.name)
        if dt is None:
            logging.warning("Skipping invalid filename: %s", path.name)
            continue

        images_by_date.setdefault(dt.date(), []).append(
            ScannedImage(name=path.name, dt=dt)
        )

    return images_by_date


def classify_images(images: List[ScannedImage]) -> List[ImageRecord]:
    """
    Sort images chronologically (ascending) and map them to public payload.
    Last image -> "0h", second-last -> "-1h", third-last -> "-2h", others -> "-2h".
    """
    sorted_images = sorted(images, key=lambda s: s.dt)

    def make_record(idx: int, img: ScannedImage, total: int) -> ImageRecord:
        if idx == total - 1:
            img_type = "0h"
        elif idx == total - 2:
            img_type = "-1h"
        elif idx == total - 3:
            img_type = "-2h"
        else:
            img_type = "-2h"
        return ImageRecord(name=img.name, datetime=img.dt.isoformat(), type=img_type)

    return [
        make_record(i, img, len(sorted_images)) for i, img in enumerate(sorted_images)
    ]


def load_existing(output_path: Path) -> Dict[str, DateEntry]:
    """
    Load existing JSON and return a dict keyed by ISO date ('YYYY-MM-DD') to DateEntry.
    If the file doesn't exist or can't be parsed, return an empty mapping.
    """
    existing_by_date: Dict[str, DateEntry] = {}
    if not output_path.exists():
        return existing_by_date

    try:
        raw = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning(
            "Could not read existing JSON (%s). A new file will be created instead.",
            exc,
        )
        return existing_by_date
    except OSError as exc:
        logging.warning(
            "Could not open existing JSON (%s). A new file will be created instead.",
            exc,
        )
        return existing_by_date

    if not isinstance(raw, list):
        logging.warning("Existing JSON not a list; ignoring and recreating.")
        return existing_by_date

    for entry in raw:
        if isinstance(entry, dict) and "date" in entry:
            date_iso = str(entry["date"])
            images = (
                entry.get("images", [])
                if isinstance(entry.get("images", []), list)
                else []
            )
            score = entry.get("score", -1)
            norm_images: List[ImageRecord] = []
            for img in images:
                if not isinstance(img, dict):
                    continue
                name = str(img.get("name", ""))
                dt_str = str(img.get("datetime", ""))
                typ = str(img.get("type", "-2h"))
                if name and dt_str:
                    norm_images.append(
                        ImageRecord(name=name, datetime=dt_str, type=typ)
                    )
            existing_by_date[date_iso] = DateEntry(
                date=date_iso, score=int(score), images=norm_images
            )

    return existing_by_date


def merge_into_existing(
    existing_by_date: Dict[str, DateEntry],
    newly_scanned: Dict[date, List[ScannedImage]],
) -> None:
    """
    Merge new scan results into existing_by_date in-place.
    - Preserve existing scores.
    - Add new dates with score = -1.
    - Ensure every entry has a 'score' field.
    - Avoid duplicate images by file name.
    - Keep images chronologically sorted by ISO datetime string.
    """
    for day, images in sorted(newly_scanned.items()):
        date_iso = day.isoformat()
        new_images_payload = classify_images(images)

        if date_iso in existing_by_date:
            entry = existing_by_date[date_iso]
            existing_names = {img["name"] for img in entry.get("images", [])}
            for img in new_images_payload:
                if img["name"] not in existing_names:
                    entry.setdefault("images", []).append(img)  # type: ignore[index]
            entry["images"].sort(key=lambda i: i["datetime"])
            entry["score"] = entry.get("score", -1)
        else:
            existing_by_date[date_iso] = DateEntry(
                date=date_iso,
                score=-1,
                images=new_images_payload,
            )

    for entry in existing_by_date.values():
        entry["score"] = entry.get("score", -1)


def to_ordered_list(existing_by_date: Dict[str, DateEntry]) -> List[DateEntry]:
    """Convert dict keyed by date to a chronologically ordered list (ascending)."""
    return [existing_by_date[k] for k in sorted(existing_by_date.keys())]


def write_json(output_path: Path, data: List[DateEntry]) -> None:
    """Write the final JSON to disk with pretty indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------- runner -----------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_dir = script_base_dir()
    img_dir = base_dir / IMG_SUBDIR
    output_path = base_dir / JSON_NAME

    logging.info("Script directory (base): %s", base_dir)
    logging.info("Images directory: %s", img_dir)
    logging.info("Output JSON path: %s", output_path)

    if not img_dir.exists():
        logging.error(
            "Image folder does not exist:\n  %s\n\n"
            "If this script is at .../sunsets/order_images.py, "
            'place your JPGs in ".../sunsets/img/ob_hotel/".',
            img_dir,
        )
        return 1
    if not img_dir.is_dir():
        logging.error('Image path is not a directory: "%s"', img_dir)
        return 1

    logging.info("Scanning images...")
    scanned = scan_images(img_dir)

    logging.info("Loading existing JSON (if present)...")
    existing_by_date = load_existing(output_path)

    logging.info("Merging and normalizing entries…")
    merge_into_existing(existing_by_date, scanned)

    final_result = to_ordered_list(existing_by_date)

    write_json(output_path, final_result)
    print(f"JSON saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
