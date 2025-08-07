import os
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# ----------------- user-configurable section -----------------
script_dir = Path(__file__).parent
img_folder = os.path.join(script_dir, "img", "ob_hotel")
json_name   = "sunset_images_grouped.json"          # Output file name
# -------------------------------------------------------------

output_path = os.path.join(script_dir, json_name)

# ---------------------------------------------------------------------------
# 1. Scan the folder and group images by calendar day
# ---------------------------------------------------------------------------
images_by_date = defaultdict(list)

for filename in os.listdir(img_folder):
    if not filename.lower().endswith(".jpg"):
        continue

    try:
        name_part = filename[:-4]           # strip '.jpg'
        date_str, time_str = name_part.split("--")
        dt = datetime.strptime(date_str + time_str, "%y_%m_%d%H_%M_%S")

        images_by_date[dt.date()].append(
            {
                "name": filename,
                "datetime": dt.isoformat(),
                "dt_obj": dt,              # keep a datetime object for sorting
            }
        )
    except Exception as exc:
        print(f"Skipping invalid filename: {filename} ({exc})")

# ---------------------------------------------------------------------------
# 2. Load existing JSON (if any) so we never clobber the scores you added
# ---------------------------------------------------------------------------
existing_by_date = {}

if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
            # Create lookup keyed by ISO-date string
            for entry in existing_data:
                if isinstance(entry, dict) and "date" in entry:
                    existing_by_date[entry["date"]] = entry
        except json.JSONDecodeError as exc:
            print(f"⚠️  Could not read existing JSON ({exc}). "
                  "A new file will be created instead.")

# ---------------------------------------------------------------------------
# 3. Merge new findings into the existing structure
#    • Keep your existing score intact
#    • Add new dates with score = −1
#    • Ensure every entry ends up with a 'score' field
# ---------------------------------------------------------------------------
for date, images in sorted(images_by_date.items()):
    date_iso = date.isoformat()

    # Sort images by time (ascending) so the last item is the most recent
    sorted_images = sorted(images, key=lambda x: x["dt_obj"])

    # Transform to the public shape (drop the helper dt_obj)
    def _img_payload(idx, img, total):
        if idx == total - 1:
            img_type = "0h"
        elif idx == total - 2:
            img_type = "-1h"
        elif idx == total - 3:
            img_type = "-2h"
        else:
            img_type = "-2h"
        return {"name": img["name"], "datetime": img["datetime"], "type": img_type}

    new_images_payload = [
        _img_payload(i, img, len(sorted_images)) for i, img in enumerate(sorted_images)
    ]

    if date_iso in existing_by_date:
        # --- Update existing entry -------------------------------------------------
        entry = existing_by_date[date_iso]

        # Merge image lists, avoiding duplicates by file name
        existing_names = {img["name"] for img in entry.get("images", [])}
        for img in new_images_payload:
            if img["name"] not in existing_names:
                entry.setdefault("images", []).append(img)

        # Ensure images remain sorted chronologically
        entry["images"].sort(key=lambda i: i["datetime"])

        # Ensure score field is present (keep original value if it exists)
        entry["score"] = entry.get("score", -1)
    else:
        # --- Brand-new date entry --------------------------------------------------
        existing_by_date[date_iso] = {
            "date": date_iso,
            "score": -1,  # mark for manual scoring later
            "images": new_images_payload,
        }

# ---------------------------------------------------------------------------
# 4. Ensure every entry (including ones that existed only in the JSON file)
#    has a score field; default to −1 so you can spot missing values
# ---------------------------------------------------------------------------
for entry in existing_by_date.values():
    entry["score"] = entry.get("score", -1)

# ---------------------------------------------------------------------------
# 5. Write the merged structure back to disk (ordered chronologically)
# ---------------------------------------------------------------------------
final_result = [
    existing_by_date[key] for key in sorted(existing_by_date.keys())
]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, indent=4)

print(f"JSON saved to: {output_path}")
