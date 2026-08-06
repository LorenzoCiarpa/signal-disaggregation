"""Schema v3 del questionario NILM: catalogo, istanze device, export JSON/Excel.

Rispetto allo schema v2 (device_usage_by_imei_v2/):
  - `devices` e' una LISTA di istanze indipendenti, non un dict per tipologia:
    due climatizzatori possono avere fasce orarie e consumi diversi.
  - si raccolgono le potenze dichiarate dall'utente (p_min/p_typical/p_max_w).
  - si raccolgono potenza contrattuale e carico base della casa.
  - ogni istanza porta un livello di confidenza, usabile per pesare i vincoli
    come soft invece che hard.
Sono state rimosse tutte le domande su abitazione, occupanti e presenza in casa,
oltre alle note in testo libero.

Questo modulo e' autonomo: non importa nulla da scripts/ e non modifica il
formato v2 letto da scripts/nilm/devices.py.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

SCHEMA_VERSION = 3

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog", "device_catalog.json")

CONFIDENCE_LEVELS = ["alta", "media", "bassa"]

MONTHS = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre",
]


def load_catalog(path: str = CATALOG_PATH) -> dict:
    """Load the device catalog (types, default powers, model stubs)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def catalog_by_type(catalog: dict) -> dict[str, dict]:
    """Index the catalog device types by their `type` name."""
    return {entry["type"]: entry for entry in catalog["device_types"]}


def lookup_model(catalog: dict, device_type: str, brand: str, model: str) -> Optional[dict]:
    """Return the catalog entry for a brand/model, or None if not found.

    Placeholder for a future real product database (energy labels, manufacturer
    APIs): the app only depends on this signature, not on the JSON stub.
    """
    entry = catalog_by_type(catalog).get(device_type)
    if entry is None:
        return None
    for candidate in entry.get("models", []):
        if candidate["brand"] == brand and candidate["model"] == model:
            return candidate
    return None


@dataclass
class DeviceInstance:
    """One physical appliance declared by the household."""

    id: str
    type: str
    label: str = ""
    count: int = 1

    brand: Optional[str] = None
    model: Optional[str] = None
    power_source: str = "catalogo"  # "catalogo" | "dichiarato" | "default"

    p_typical_w: Optional[float] = None
    p_min_w: Optional[float] = None
    p_max_w: Optional[float] = None

    usage_mode: str = "cycle"  # cycle | daily_hours | always_on

    # usage_mode == "cycle"
    usage_frequency_per_week_min: Optional[float] = None
    usage_frequency_per_week_max: Optional[float] = None
    duration_minutes_min: Optional[float] = None
    duration_minutes_max: Optional[float] = None

    # usage_mode == "daily_hours"
    daily_usage_hours_min: Optional[float] = None
    daily_usage_hours_max: Optional[float] = None

    # finestra oraria, in minuti dalla mezzanotte
    start_window_start_min: Optional[int] = None
    start_window_end_min: Optional[int] = None
    start_window_wraps_next_day: bool = False

    active_months: Optional[list[int]] = None
    confidence: str = "media"


@dataclass
class Household:
    """A full questionnaire response for one metered household."""

    imei: str
    contract_power_kw: Optional[float] = None
    base_load_w: Optional[float] = None
    base_load_confidence: str = "media"
    devices: list[DeviceInstance] = field(default_factory=list)


def _sorted_months(months: Optional[list[int]]) -> Optional[list[int]]:
    if not months:
        return None
    return sorted(set(int(m) for m in months))


def validate(household: Household) -> list[str]:
    """Return a list of human-readable problems; empty means the response is valid."""
    problems: list[str] = []

    if not household.imei.strip():
        problems.append("IMEI mancante.")
    elif not household.imei.strip().isdigit():
        problems.append("L'IMEI deve contenere solo cifre.")

    if household.contract_power_kw is None:
        problems.append("Potenza contrattuale (kW) mancante.")
    elif household.contract_power_kw <= 0:
        problems.append("La potenza contrattuale deve essere maggiore di zero.")

    if not household.devices:
        problems.append("Nessun dispositivo inserito.")

    seen_labels: set[tuple[str, str]] = set()
    for device in household.devices:
        who = f"{device.type}" + (f" ({device.label})" if device.label else "")

        # Due dispositivi dello stesso tipo senza etichetta sono legittimi; il
        # conflitto esiste solo se entrambi hanno la STESSA etichetta esplicita.
        label = device.label.strip().lower()
        if label:
            key = (device.type, label)
            if key in seen_labels:
                problems.append(
                    f"{who}: due dispositivi dello stesso tipo hanno la stessa etichetta."
                )
            seen_labels.add(key)

        if device.count < 1:
            problems.append(f"{who}: la quantita' deve essere almeno 1.")

        lo, typ, hi = device.p_min_w, device.p_typical_w, device.p_max_w
        if typ is not None:
            if lo is not None and lo > typ:
                problems.append(f"{who}: potenza minima maggiore della tipica.")
            if hi is not None and hi < typ:
                problems.append(f"{who}: potenza massima minore della tipica.")
        if lo is not None and hi is not None and lo > hi:
            problems.append(f"{who}: potenza minima maggiore della massima.")

        pairs = [
            ("frequenza settimanale", device.usage_frequency_per_week_min, device.usage_frequency_per_week_max),
            ("durata del ciclo", device.duration_minutes_min, device.duration_minutes_max),
            ("ore giornaliere", device.daily_usage_hours_min, device.daily_usage_hours_max),
        ]
        for name, low, high in pairs:
            if low is not None and high is not None and low > high:
                problems.append(f"{who}: {name}, il minimo supera il massimo.")

        if device.daily_usage_hours_max is not None and device.daily_usage_hours_max > 24:
            problems.append(f"{who}: le ore giornaliere non possono superare 24.")

        if device.confidence not in CONFIDENCE_LEVELS:
            problems.append(f"{who}: livello di confidenza non valido.")

    return problems


def to_dict(household: Household) -> dict:
    """Serialize a household response to the schema v3 JSON structure."""
    devices = []
    for device in household.devices:
        payload = asdict(device)
        payload["active_months"] = _sorted_months(device.active_months)
        devices.append(payload)

    return {
        "schema_version": SCHEMA_VERSION,
        "imei": household.imei.strip(),
        "household": {
            "contract_power_kw": household.contract_power_kw,
            "base_load_w": household.base_load_w,
            "base_load_confidence": household.base_load_confidence,
        },
        "devices": devices,
    }


def to_json(household: Household) -> str:
    return json.dumps(to_dict(household), indent=2, ensure_ascii=False)


# Colonne dell'export tabellare: una riga per istanza di dispositivo.
EXCEL_COLUMNS = [
    "imei",
    "contract_power_kw",
    "base_load_w",
    "base_load_confidence",
    "device_id",
    "device_type",
    "label",
    "count",
    "brand",
    "model",
    "power_source",
    "p_min_w",
    "p_typical_w",
    "p_max_w",
    "usage_mode",
    "usage_frequency_per_week_min",
    "usage_frequency_per_week_max",
    "duration_minutes_min",
    "duration_minutes_max",
    "daily_usage_hours_min",
    "daily_usage_hours_max",
    "start_window_start_min",
    "start_window_end_min",
    "start_window_wraps_next_day",
    "active_months",
    "confidence",
]


def to_rows(household: Household) -> list[dict]:
    """Flatten a household response to one row per device instance."""
    rows = []
    for device in household.devices:
        months = _sorted_months(device.active_months)
        rows.append({
            "imei": household.imei.strip(),
            "contract_power_kw": household.contract_power_kw,
            "base_load_w": household.base_load_w,
            "base_load_confidence": household.base_load_confidence,
            "device_id": device.id,
            "device_type": device.type,
            "label": device.label,
            "count": device.count,
            "brand": device.brand,
            "model": device.model,
            "power_source": device.power_source,
            "p_min_w": device.p_min_w,
            "p_typical_w": device.p_typical_w,
            "p_max_w": device.p_max_w,
            "usage_mode": device.usage_mode,
            "usage_frequency_per_week_min": device.usage_frequency_per_week_min,
            "usage_frequency_per_week_max": device.usage_frequency_per_week_max,
            "duration_minutes_min": device.duration_minutes_min,
            "duration_minutes_max": device.duration_minutes_max,
            "daily_usage_hours_min": device.daily_usage_hours_min,
            "daily_usage_hours_max": device.daily_usage_hours_max,
            "start_window_start_min": device.start_window_start_min,
            "start_window_end_min": device.start_window_end_min,
            "start_window_wraps_next_day": device.start_window_wraps_next_day,
            "active_months": ",".join(str(m) for m in months) if months else None,
            "confidence": device.confidence,
        })
    return rows


def minutes_to_hhmm(minutes: Optional[int]) -> str:
    if minutes is None:
        return "--:--"
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
