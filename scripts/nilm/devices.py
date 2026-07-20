"""
US-002: Knowledge base dispositivi e inventario per IMEI.

Provides the DeviceProfile dataclass, DEVICE_KNOWLEDGE_BASE, and
get_device_profiles() to load per-IMEI survey device inventories.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceProfile:
    """Typical power/usage profile for a household appliance."""

    name: str
    p_min_w: float
    p_typical_w: float
    p_max_w: float
    dur_min_min: float
    dur_typical_min: float
    duty_cycle: Optional[float]
    frequency_per_week: float
    prior_weight: float
    always_on: bool = False
    frequency_per_week_min: Optional[float] = None
    frequency_per_week_max: Optional[float] = None
    duration_minutes_max: Optional[float] = None
    start_window_start_min: Optional[int] = None
    start_window_end_min: Optional[int] = None
    start_window_wraps_next_day: bool = False
    daily_usage_hours_min: Optional[float] = None
    daily_usage_hours_max: Optional[float] = None
    active_months: Optional[list[int]] = None


DEVICE_KNOWLEDGE_BASE: dict[str, dict] = {
    "Congelatore": {
        "p_min_w": 30.0,
        "p_typical_w": 100.0,
        "p_max_w": 200.0,
        "dur_min_min": 5.0,
        "dur_typical_min": 20.0,
        "duty_cycle": 0.25,
        "frequency_per_week": 7.0,
        "always_on": True,
    },
    "Lavatrice": {
        "p_min_w": 300.0,
        "p_typical_w": 1800.0,
        "p_max_w": 3000.0,
        "dur_min_min": 30.0,
        "dur_typical_min": 90.0,
        "duty_cycle": None,
        "frequency_per_week": 3.0,
    },
    "Asciugatrice": {
        "p_min_w": 500.0,
        "p_typical_w": 2500.0,
        "p_max_w": 4000.0,
        "dur_min_min": 30.0,
        "dur_typical_min": 75.0,
        "duty_cycle": None,
        "frequency_per_week": 2.0,
    },
    "Lavastoviglie": {
        "p_min_w": 200.0,
        "p_typical_w": 1500.0,
        "p_max_w": 2500.0,
        "dur_min_min": 45.0,
        "dur_typical_min": 120.0,
        "duty_cycle": None,
        "frequency_per_week": 3.0,
    },
    "Forno": {
        "p_min_w": 500.0,
        "p_typical_w": 2500.0,
        "p_max_w": 4000.0,
        "dur_min_min": 10.0,
        "dur_typical_min": 45.0,
        "duty_cycle": None,
        "frequency_per_week": 3.0,
    },
    "Piano cottura": {
        "p_min_w": 200.0,
        "p_typical_w": 1500.0,
        "p_max_w": 3000.0,
        "dur_min_min": 5.0,
        "dur_typical_min": 30.0,
        "duty_cycle": None,
        "frequency_per_week": 7.0,
    },
    "Microonde": {
        "p_min_w": 300.0,
        "p_typical_w": 900.0,
        "p_max_w": 1500.0,
        "dur_min_min": 1.0,
        "dur_typical_min": 5.0,
        "duty_cycle": None,
        "frequency_per_week": 7.0,
    },
    "Boiler": {
        "p_min_w": 500.0,
        "p_typical_w": 2000.0,
        "p_max_w": 3000.0,
        "dur_min_min": 10.0,
        "dur_typical_min": 45.0,
        "duty_cycle": None,
        "frequency_per_week": 7.0,
    },
    "Climatizzatore": {
        "p_min_w": 300.0,
        "p_typical_w": 1500.0,
        "p_max_w": 3000.0,
        "dur_min_min": 15.0,
        "dur_typical_min": 90.0,
        "duty_cycle": None,
        "frequency_per_week": 5.0,
    },
    "Televisori": {
        "p_min_w": 30.0,
        "p_typical_w": 100.0,
        "p_max_w": 300.0,
        "dur_min_min": 15.0,
        "dur_typical_min": 120.0,
        "duty_cycle": None,
        "frequency_per_week": 7.0,
    },
    "Computer": {
        "p_min_w": 30.0,
        "p_typical_w": 150.0,
        "p_max_w": 400.0,
        "dur_min_min": 15.0,
        "dur_typical_min": 180.0,
        "duty_cycle": None,
        "frequency_per_week": 5.0,
    },
    "Console": {
        "p_min_w": 50.0,
        "p_typical_w": 150.0,
        "p_max_w": 300.0,
        "dur_min_min": 15.0,
        "dur_typical_min": 120.0,
        "duty_cycle": None,
        "frequency_per_week": 3.0,
    },
    "Auto elettrica": {
        "p_min_w": 1000.0,
        "p_typical_w": 3700.0,
        "p_max_w": 11000.0,
        "dur_min_min": 60.0,
        "dur_typical_min": 360.0,
        "duty_cycle": None,
        "frequency_per_week": 3.0,
    },
    "Frigorifero principale": {
        "p_min_w": 50.0,
        "p_typical_w": 125.0,
        "p_max_w": 300.0,
        "dur_min_min": 5.0,
        "dur_typical_min": 20.0,
        "duty_cycle": 0.30,
        "frequency_per_week": 7.0,
        "always_on": True,
    },
    "Frigorifero secondario": {
        "p_min_w": 30.0,
        "p_typical_w": 100.0,
        "p_max_w": 200.0,
        "dur_min_min": 5.0,
        "dur_typical_min": 20.0,
        "duty_cycle": 0.25,
        "frequency_per_week": 7.0,
        "always_on": True,
    },
}


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _representative_from_bounds(
    lower: Optional[float], upper: Optional[float], fallback: float
) -> float:
    if lower is not None and upper is not None:
        return (lower + upper) / 2.0
    if lower is not None:
        return max(lower, fallback)
    if upper is not None:
        return min(upper, fallback)
    return fallback


def get_device_profiles(
    imei: str, device_dir: str = "device_usage_by_imei_v2"
) -> list[DeviceProfile]:
    """Load survey-aware device profiles for a given IMEI from the inventory JSON.

    Devices marked present=true get prior_weight=1.0; absent devices get 0.05.
    Survey bounds (usage frequency, duration, start window, daily hours, active
    months) override the DEVICE_KNOWLEDGE_BASE defaults where present.

    Args:
        imei: The IMEI string identifying the household.
        device_dir: Directory containing per-IMEI JSON files.

    Returns:
        List of DeviceProfile instances for the supported device set.
    """
    path = os.path.join(device_dir, f"{imei}.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    devices_data = data.get("devices", {})
    profiles: list[DeviceProfile] = []

    for device_name, kb_entry in DEVICE_KNOWLEDGE_BASE.items():
        device_info = devices_data.get(device_name, {})
        present = bool(device_info.get("present", False))
        prior_weight = 1.0 if present else 0.05

        frequency_per_week_min = _safe_float(device_info.get("usage_frequency_per_week_min"))
        frequency_per_week_max = _safe_float(device_info.get("usage_frequency_per_week_max"))
        duration_minutes_min = _safe_float(device_info.get("duration_minutes_min"))
        duration_minutes_max = _safe_float(device_info.get("duration_minutes_max"))
        start_window_start_min = _safe_int(device_info.get("start_window_start_min"))
        start_window_end_min = _safe_int(device_info.get("start_window_end_min"))
        daily_usage_hours_min = _safe_float(device_info.get("daily_usage_hours_min"))
        daily_usage_hours_max = _safe_float(device_info.get("daily_usage_hours_max"))

        active_months_raw = device_info.get("active_months")
        active_months = None
        if isinstance(active_months_raw, list):
            active_months = [int(month) for month in active_months_raw]

        frequency_per_week = _representative_from_bounds(
            frequency_per_week_min,
            frequency_per_week_max,
            kb_entry["frequency_per_week"],
        )
        dur_min_min = kb_entry["dur_min_min"]
        if duration_minutes_min is not None and duration_minutes_min > 0:
            dur_min_min = duration_minutes_min
        dur_typical_min = _representative_from_bounds(
            duration_minutes_min,
            duration_minutes_max,
            kb_entry["dur_typical_min"],
        )
        dur_typical_min = max(dur_typical_min, dur_min_min)

        profile = DeviceProfile(
            name=device_name,
            p_min_w=kb_entry["p_min_w"],
            p_typical_w=kb_entry["p_typical_w"],
            p_max_w=kb_entry["p_max_w"],
            dur_min_min=dur_min_min,
            dur_typical_min=dur_typical_min,
            duty_cycle=kb_entry["duty_cycle"],
            frequency_per_week=frequency_per_week,
            prior_weight=prior_weight,
            always_on=kb_entry.get("always_on", False),
            frequency_per_week_min=frequency_per_week_min,
            frequency_per_week_max=frequency_per_week_max,
            duration_minutes_max=duration_minutes_max,
            start_window_start_min=start_window_start_min,
            start_window_end_min=start_window_end_min,
            start_window_wraps_next_day=bool(
                device_info.get("start_window_wraps_next_day", False)
            ),
            daily_usage_hours_min=daily_usage_hours_min,
            daily_usage_hours_max=daily_usage_hours_max,
            active_months=active_months,
        )
        profiles.append(profile)

    return profiles


if __name__ == "__main__":
    from scripts.nilm.preprocessing import get_usable_imeis

    for imei in get_usable_imeis():
        profiles = get_device_profiles(imei)
        present = [p.name for p in profiles if p.prior_weight >= 1.0]
        print(f"IMEI {imei}: {len(present)} devices present — {present}")
