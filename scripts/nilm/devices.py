"""
US-002: Knowledge base dispositivi e inventario per IMEI.

Provides DeviceProfile dataclass, DEVICE_KNOWLEDGE_BASE, parse_frequency(),
and get_device_profiles() to load per-IMEI device inventories.
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

_FREQUENCY_MAP: dict[str, float] = {
    "quasi ogni giorno": 6.0,
    "più di 7 volte": 8.0,
    "1-2 volte": 1.5,
    "pranzo": 7.0,
    "notte": 7.0,
}


def parse_frequency(s: Optional[str]) -> float:
    """Map an Italian frequency string to a weekly frequency float.

    Args:
        s: Italian frequency string from device survey, or None/empty.

    Returns:
        Weekly frequency as float. Defaults to 3.0 for unknown/empty values.
    """
    if not s:
        return 3.0
    normalized = s.strip().lower()
    return _FREQUENCY_MAP.get(normalized, 3.0)


def get_device_profiles(
    imei: str, device_dir: str = "device_usage_by_imei"
) -> list[DeviceProfile]:
    """Load device profiles for a given IMEI from the inventory JSON.

    Devices marked present=true get prior_weight=1.0; absent devices get 0.05.
    Frequency comes from parse_frequency() applied to the survey field.

    Args:
        imei: The IMEI string identifying the household.
        device_dir: Directory containing per-IMEI JSON files.

    Returns:
        List of DeviceProfile instances for all modeled devices.
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
        frequency_str = device_info.get("frequency", "") if device_info else ""
        frequency_per_week = parse_frequency(frequency_str)

        profile = DeviceProfile(
            name=device_name,
            p_min_w=kb_entry["p_min_w"],
            p_typical_w=kb_entry["p_typical_w"],
            p_max_w=kb_entry["p_max_w"],
            dur_min_min=kb_entry["dur_min_min"],
            dur_typical_min=kb_entry["dur_typical_min"],
            duty_cycle=kb_entry["duty_cycle"],
            frequency_per_week=frequency_per_week,
            prior_weight=prior_weight,
            always_on=kb_entry.get("always_on", False),
        )
        profiles.append(profile)

    return profiles


if __name__ == "__main__":
    from scripts.nilm.preprocessing import get_usable_imeis

    for imei in get_usable_imeis():
        profiles = get_device_profiles(imei)
        present = [p.name for p in profiles if p.prior_weight >= 1.0]
        print(f"IMEI {imei}: {len(present)} devices present — {present}")
