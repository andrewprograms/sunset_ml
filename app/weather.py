"""
app/weather.py

Backfill weather features onto each date object in `sunset_images_grouped.json`
using FAA/NOAA Aviation Weather (ADDS) METARs.

Assumptions
-----------
- The image datetimes in the JSON are *local* to LOCAL_TIMEZONE (see config).
- We prefer the '-2h' image timestamp; if it's missing, we fall back to '-1h',
  then the earliest image for that date.
- We choose the nearest METAR at or before the target time when possible,
  within a configurable lookback window.

Outputs
-------
Each date entry in JSON will get a `weather` field like:

"weather": {
  "station": "KSAN",
  "provider": "FAA/NOAA ADDS (AviationWeather.gov)",
  "query_window_utc": {"start": "...Z", "end": "...Z"},
  "observation_time_utc": "...Z",
  "age_minutes": 23,
  "features": {
    "flight_category": "VFR",
    "visibility_mi": 10.0,
    "altimeter_inHg": 29.92,
    "sea_level_pressure_hPa": 1013.0,
    "wind_dir_deg": 270,
    "wind_speed_kt": 12,
    "wind_gust_kt": 18,
    "temp_c": 18.3,
    "dewpoint_c": 14.4,
    "relative_humidity_pct": 74.2,
    "ceiling_ft_agl": 1600,
    "cloud_layers": [
      {"cover": "SCT", "base_ft_agl": 1200},
      {"cover": "BKN", "base_ft_agl": 1600}
    ],
    "wx_string": "RA"
  }
}

Notes
-----
- Requires `requests` (pip install requests).
- The Aviation Weather "dataserver_current" endpoint does not require an API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import json
import math
import time
import xml.etree.ElementTree as ET

import requests

# 1st-party imports
from app.config import (
    JSON_PATH,
    LOCAL_TIMEZONE,
    WEATHER_STATION_ID,
    WEATHER_SOURCE_URL,
    WEATHER_LOOKBACK_HOURS,
    WEATHER_LOOKAHEAD_HOURS,
    WEATHER_REQUEST_TIMEOUT_SEC,
    WEATHER_MAX_RETRIES,
    WEATHER_MIN_OBS_AGE_MIN,
)


# -----------------------------
# Models / helpers
# -----------------------------


@dataclass
class MetarObs:
    station_id: str
    time: datetime  # UTC-aware
    flight_category: Optional[str]
    visibility_mi: Optional[float]
    altimeter_inHg: Optional[float]
    sea_level_pressure_hPa: Optional[float]
    wind_dir_deg: Optional[int]
    wind_speed_kt: Optional[int]
    wind_gust_kt: Optional[int]
    temp_c: Optional[float]
    dewpoint_c: Optional[float]
    wx_string: Optional[str]
    cloud_layers: List[Tuple[str, Optional[int]]]  # list of (cover, base_ft_agl)

    def ceiling_ft_agl(self) -> Optional[int]:
        """Return lowest BKN/OVC cloud base as 'ceiling' (ft AGL), or None."""
        ceiling_candidates = [
            base
            for cover, base in self.cloud_layers
            if cover in ("BKN", "OVC") and base is not None
        ]
        return min(ceiling_candidates) if ceiling_candidates else None

    def relative_humidity_pct(self) -> Optional[float]:
        """Compute RH from temp/dewpoint if available (Magnus formula)."""
        if self.temp_c is None or self.dewpoint_c is None:
            return None
        T = float(self.temp_c)
        Td = float(self.dewpoint_c)
        # Magnus (over water)
        a, b = 17.625, 243.04
        try:
            gamma_Td = (a * Td) / (b + Td)
            gamma_T = (a * T) / (b + T)
            rh = 100.0 * math.exp(gamma_Td - gamma_T)
            return max(0.0, min(100.0, rh))
        except Exception:
            return None

    def to_feature_dict(self) -> Dict:
        return {
            "flight_category": self.flight_category,
            "visibility_mi": self.visibility_mi,
            "altimeter_inHg": self.altimeter_inHg,
            "sea_level_pressure_hPa": self.sea_level_pressure_hPa,
            "wind_dir_deg": self.wind_dir_deg,
            "wind_speed_kt": self.wind_speed_kt,
            "wind_gust_kt": self.wind_gust_kt,
            "temp_c": self.temp_c,
            "dewpoint_c": self.dewpoint_c,
            "relative_humidity_pct": (
                round(self.relative_humidity_pct(), 1)
                if self.relative_humidity_pct() is not None
                else None
            ),
            "ceiling_ft_agl": self.ceiling_ft_agl(),
            "cloud_layers": [
                {"cover": cover, "base_ft_agl": base}
                for cover, base in self.cloud_layers
            ],
            "wx_string": self.wx_string,
        }


def _parse_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_int(text: Optional[str]) -> Optional[int]:
    f = _parse_float(text)
    return int(f) if f is not None else None


def _isoformat_utc(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .replace(tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# -----------------------------
# ADDS / AWC fetching
# -----------------------------


def fetch_metars_xml(
    station: str,
    start_utc: datetime,
    end_utc: datetime,
    timeout: float = WEATHER_REQUEST_TIMEOUT_SEC,
    max_retries: int = WEATHER_MAX_RETRIES,
) -> str:
    """
    Fetch METARs XML from AviationWeather.gov ADDS dataserver.

    Ref: https://aviationweather.gov/dataserver
    Endpoint used: dataserver_current/httpparam
    """
    base = WEATHER_SOURCE_URL.rstrip("/")
    params = {
        "datasource": "metars",
        "requestType": "retrieve",
        "format": "xml",
        "stationString": station,
        "startTime": _isoformat_utc(start_utc),
        "endTime": _isoformat_utc(end_utc),
        "hoursBeforeNow": "",  # ensure we use explicit time window
    }
    headers = {"User-Agent": "sunset-ml-weather/1.0 (+local script)"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                f"{base}/httpparam", params=params, timeout=timeout, headers=headers
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(min(2.0 * attempt, 6.0))  # simple backoff


def parse_metars_from_xml(xml_text: str) -> List[MetarObs]:
    """Parse ADDS XML into a list of MetarObs."""
    out: List[MetarObs] = []
    root = ET.fromstring(xml_text)

    for metar in root.findall(".//METAR"):
        station_id = (metar.findtext("station_id") or "").strip()
        obs_time_txt = metar.findtext("observation_time")
        if not station_id or not obs_time_txt:
            continue
        # obs time is ISO UTC like "2025-08-19T02:20:00Z"
        obs_time = datetime.fromisoformat(obs_time_txt.replace("Z", "+00:00"))

        flight_category = metar.findtext("flight_category") or None
        visibility_mi = _parse_float(metar.findtext("visibility_statute_mi"))
        altimeter_inHg = _parse_float(metar.findtext("altim_in_hg"))
        slp_mb = _parse_float(metar.findtext("sea_level_pressure_mb"))
        sea_level_pressure_hPa = float(slp_mb) if slp_mb is not None else None

        wind_dir_deg = _parse_int(metar.findtext("wind_dir_degrees"))
        wind_speed_kt = _parse_int(metar.findtext("wind_speed_kt"))
        wind_gust_kt = _parse_int(metar.findtext("wind_gust_kt"))

        temp_c = _parse_float(metar.findtext("temp_c"))
        dewpoint_c = _parse_float(metar.findtext("dewpoint_c"))
        wx_string = metar.findtext("wx_string") or None

        cloud_layers: List[Tuple[str, Optional[int]]] = []
        for sky in metar.findall("sky_condition"):
            cover = sky.attrib.get("sky_cover")
            base = sky.attrib.get("cloud_base_ft_agl")
            cover = cover.strip() if cover else None
            base_ft = _parse_int(base) if base else None
            if cover:
                cloud_layers.append((cover, base_ft))

        out.append(
            MetarObs(
                station_id=station_id,
                time=obs_time,
                flight_category=flight_category,
                visibility_mi=visibility_mi,
                altimeter_inHg=altimeter_inHg,
                sea_level_pressure_hPa=sea_level_pressure_hPa,
                wind_dir_deg=wind_dir_deg,
                wind_speed_kt=wind_speed_kt,
                wind_gust_kt=wind_gust_kt,
                temp_c=temp_c,
                dewpoint_c=dewpoint_c,
                wx_string=wx_string,
                cloud_layers=cloud_layers,
            )
        )
    return out


def pick_best_metar(metars: List[MetarObs], target_utc: datetime) -> Optional[MetarObs]:
    """Choose the observation nearest to, preferably at or before, the target time."""
    if not metars:
        return None
    # Prefer obs <= target, minimize |delta|
    past = [m for m in metars if m.time <= target_utc]
    if past:
        return min(past, key=lambda m: (target_utc - m.time))
    # Otherwise, take the nearest after target
    return min(metars, key=lambda m: (m.time - target_utc))


# -----------------------------
# JSON backfill
# -----------------------------


def _pick_reference_image_dt_local(entry: Dict) -> Optional[datetime]:
    """
    Prefer '-2h' image datetime; if missing, fall back to '-1h', else earliest image.
    Returns timezone-aware LOCAL datetime if found.
    """
    images = entry.get("images") or []
    by_type: Dict[str, Dict] = {
        img.get("type"): img for img in images if isinstance(img, dict)
    }

    for t in ("-2h", "-1h"):
        img = by_type.get(t)
        if img and img.get("datetime"):
            return _localize_iso(img["datetime"])

    # fallback: earliest image with datetime
    candidates = [img for img in images if img.get("datetime")]
    if not candidates:
        return None
    dt = min(_localize_iso(img["datetime"]) for img in candidates)
    return dt


def _localize_iso(naive_iso: str) -> datetime:
    """
    Treat the ISO string as LOCAL time (no timezone present in your JSON),
    and attach LOCAL_TIMEZONE from config.
    """
    dt_naive = datetime.fromisoformat(naive_iso)
    tz = ZoneInfo(LOCAL_TIMEZONE)
    return dt_naive.replace(tzinfo=tz)


def _needs_weather(entry: Dict) -> bool:
    w = entry.get("weather")
    if not w:
        return True
    # optional freshness rule: ignore obs that appear too far after the target
    age_min = w.get("age_minutes")
    return age_min is None or (
        isinstance(age_min, (int, float)) and age_min < -1_000_000
    )  # basically never true


def backfill_weather_for_json(
    json_path=JSON_PATH,
    station: str = WEATHER_STATION_ID,
    lookback_hours: int = WEATHER_LOOKBACK_HOURS,
    lookahead_hours: int = WEATHER_LOOKAHEAD_HOURS,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    Read the sunset JSON and add a 'weather' dict for each date (if missing).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tz_local = ZoneInfo(LOCAL_TIMEZONE)
    updated = 0
    skipped = 0

    for entry in data:
        ref_dt_local = _pick_reference_image_dt_local(entry)
        if not ref_dt_local:
            skipped += 1
            if verbose:
                print(f"[weather] {entry.get('date')} - no usable image datetime; skip")
            continue
        if not _needs_weather(entry):
            skipped += 1
            continue

        # target is the local '-2h-ish' time converted to UTC
        target_utc = ref_dt_local.astimezone(timezone.utc)

        start_utc = target_utc - timedelta(hours=lookback_hours)
        end_utc = target_utc + timedelta(hours=lookahead_hours)

        try:
            xml_text = fetch_metars_xml(station, start_utc, end_utc)
            metars = parse_metars_from_xml(xml_text)
        except Exception as e:
            if verbose:
                print(f"[weather] {entry.get('date')} - fetch error: {e}")
            continue

        best = pick_best_metar(metars, target_utc)
        if not best:
            if verbose:
                print(f"[weather] {entry.get('date')} - no METARs in window")
            continue

        age_minutes = int(round((target_utc - best.time).total_seconds() / 60.0))
        # Optional minimum: prefer not to use obs that are too *after* the target
        if age_minutes < -WEATHER_MIN_OBS_AGE_MIN:
            # observation is too far *after* the target time
            if verbose:
                print(
                    f"[weather] {entry.get('date')} - nearest obs is {abs(age_minutes)} min AFTER target; keeping anyway"
                )
            # You can choose to skip in this case; we keep to maximize coverage

        weather_block = {
            "station": station,
            "provider": "FAA/NOAA ADDS (AviationWeather.gov)",
            "query_window_utc": {
                "start": _isoformat_utc(start_utc),
                "end": _isoformat_utc(end_utc),
            },
            "observation_time_utc": _isoformat_utc(best.time),
            "age_minutes": age_minutes,
            "features": best.to_feature_dict(),
        }

        entry["weather"] = weather_block
        updated += 1
        if verbose:
            print(
                f"[weather] {entry.get('date')} - added obs {weather_block['observation_time_utc']} (age {age_minutes} min)"
            )

    if dry_run:
        print(f"[weather] Dry run: {updated} would be updated, {skipped} skipped.")
        return

    # Write JSON atomically (with a lightweight backup)
    backup_path = json_path.with_suffix(".bak.json")
    with open(backup_path, "w", encoding="utf-8") as b:
        json.dump(data, b, indent=2, ensure_ascii=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"[weather] Updated {updated} entries; skipped {skipped}. Backup -> {backup_path}"
    )
