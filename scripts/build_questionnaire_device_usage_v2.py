#!/usr/bin/env python3
"""Build normalized questionnaire outputs for device usage priors.

The script reads a raw questionnaire export in CSV or Excel form and produces:
1. A compact Excel workbook with a household summary sheet and a long device sheet.
2. A per-IMEI JSON directory with numeric bounds suitable for downstream models.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_EXCEL = Path("other_data/questionnaire_device_usage_v2.xlsx")
DEFAULT_OUTPUT_JSON_DIR = Path("device_usage_by_imei_v2")

OUTPUT_DEVICE_ORDER = [
    "Congelatore",
    "Lavatrice",
    "Asciugatrice",
    "Lavastoviglie",
    "Forno",
    "Piano cottura",
    "Microonde",
    "Boiler",
    "Climatizzatore",
    "Televisori",
    "Computer",
    "Console",
    "Auto elettrica",
    "Frigorifero principale",
    "Frigorifero secondario",
]

PRESENCE_COLUMN_LABELS = {
    "Frigorifero principale": "frigorifero principale",
    "Frigorifero secondario": "frigorifero secondario",
    "Congelatore": "congelatore",
    "Lavatrice": "lavatrice",
    "Asciugatrice": "asciugatrice",
    "Lavastoviglie": "lavastoviglie",
    "Forno": "forno elettrico",
    "Piano cottura elettrico": "piano cottura elettrico",
    "Piano cottura induzione": "piano cottura induzione",
    "Microonde": "microonde",
    "Boiler": "boiler elettrico",
    "Climatizzatore": "climatizzatore fisso",
    "Televisori": "televisori",
    "Computer": "computer / notebook",
    "Console": "console videogiochi",
    "Auto elettrica": "auto elettrica / plug in",
}

DETAIL_COLUMN_PATTERNS = {
    "lavatrice_frequency": ("lavatrice", "frequenza di utilizzo"),
    "lavatrice_time": ("lavatrice", "fasce orarie"),
    "lavatrice_duration": ("lavatrice", "durata tipica"),
    "lavastoviglie_frequency": ("lavastoviglie", "frequenza media"),
    "lavastoviglie_time": ("lavastoviglie", "orario di avvio"),
    "lavastoviglie_duration": ("lavastoviglie", "durata del ciclo"),
    "forno_frequency": ("forno elettrico", "frequenza settimanale"),
    "forno_duration": ("forno elettrico", "durata media"),
    "climate_period": ("climatizzatore / pompa di calore", "periodi di utilizzo principale"),
    "climate_daily_hours": ("climatizzatore / pompa di calore", "ore medie giornalieri"),
}

HOME_TYPE_CODE = {
    "appartamento in condominio": 1,
    "appartamento in palazzo familiare": 2,
    "mansarda in palazzo familiare": 3,
    "villetta a schiera": 4,
    "casa indipendente": 5,
}

HEATING_TYPE_CODE = {
    "caldaia a gas / metano": 1,
    "pompa di calore / climatizzatore": 2,
    "pompa di calore alimentata da fotovoltaico condominiale": 3,
}

PRESENCE_SCORE = {
    "quasi mai": 0,
    "a volte": 1,
    "spesso": 2,
    "sempre": 3,
}

WEEKEND_PATTERN_CODE = {
    "piu spesso in casa": 1,
    "piu spesso fuori casa": -1,
    "come nei giorni feriali": 0,
}

TIME_WINDOW_MAP = {
    "mattina": (6 * 60, 10 * 60, False),
    "pranzo": (12 * 60, 15 * 60, False),
    "pomeriggio": (10 * 60, 17 * 60, False),
    "sera": (17 * 60, 22 * 60, False),
    "cena": (19 * 60, 22 * 60, False),
    "notte": (22 * 60, 6 * 60, True),
}

MONTH_NAME_TO_NUM = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}


def is_missing(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value)
    text = text.replace("\xa0", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value: Any) -> str:
    text = clean_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()


class ColumnLocator:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.normalized = {column: normalize_key(column) for column in columns}

    def find(self, *terms: str) -> str | None:
        normalized_terms = [normalize_key(term) for term in terms]
        matches = [
            column
            for column, normalized in self.normalized.items()
            if all(term in normalized for term in normalized_terms)
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda column: (len(column), column))[0]

    def find_all(self, *terms: str) -> list[str]:
        normalized_terms = [normalize_key(term) for term in terms]
        matches = [
            column
            for column, normalized in self.normalized.items()
            if all(term in normalized for term in normalized_terms)
        ]
        return sorted(matches)


def empty_device_payload() -> dict[str, Any]:
    return {
        "present": False,
        "count_min": 0,
        "count_max": 0,
        "count_open_upper": False,
        "usage_frequency_per_week_min": None,
        "usage_frequency_per_week_max": None,
        "usage_frequency_per_week_open_upper": False,
        "duration_minutes_min": None,
        "duration_minutes_max": None,
        "duration_minutes_open_upper": False,
        "start_window_start_min": None,
        "start_window_end_min": None,
        "start_window_wraps_next_day": False,
        "daily_usage_hours_min": None,
        "daily_usage_hours_max": None,
        "daily_usage_hours_open_upper": False,
        "active_months": None,
        "warnings": [],
    }


def normalize_imei(value: Any) -> str | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not math.isnan(value):
        return str(int(value))
    text = clean_text(value)
    match = re.fullmatch(r"(\d+)(?:\.0+)?", text)
    if match:
        return match.group(1)
    return None


def parse_int_like(value: Any) -> int | None:
    if is_missing(value):
        return None
    text = clean_text(value)
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_binary_yes_no(value: Any) -> int | None:
    text = normalize_key(value)
    if not text:
        return None
    if text in {"si", "s", "yes", "1", "true"}:
        return 1
    if text in {"no", "0", "false", "mai"}:
        return 0
    return None


def parse_reported_flag(value: Any) -> int | None:
    text = normalize_key(value)
    if not text:
        return None
    if text in {"no", "0", "false", "mai", "nessuna", "nessuno"}:
        return 0
    return 1


def parse_count_bounds(value: Any) -> dict[str, Any]:
    text = normalize_key(value)
    if not text or text in {"non presente", "no", "false", "0"}:
        return {"count_min": 0, "count_max": 0, "count_open_upper": False, "warnings": []}
    if text in {"si", "yes", "true"}:
        return {"count_min": 1, "count_max": 1, "count_open_upper": False, "warnings": []}
    if re.fullmatch(r">\s*\d+", text):
        lower = int(re.findall(r"\d+", text)[0]) + 1
        return {
            "count_min": lower,
            "count_max": None,
            "count_open_upper": True,
            "warnings": ["count_open_upper"],
        }
    try:
        parsed = int(float(text))
    except ValueError:
        return {
            "count_min": 0,
            "count_max": 0,
            "count_open_upper": False,
            "warnings": ["count_unparsed"],
        }
    return {"count_min": parsed, "count_max": parsed, "count_open_upper": False, "warnings": []}


def combine_count_bounds(*parts: dict[str, Any]) -> dict[str, Any]:
    count_min = sum(part["count_min"] for part in parts)
    count_open_upper = any(part["count_open_upper"] for part in parts)
    count_max = None if count_open_upper else sum(part["count_max"] for part in parts)
    warnings: list[str] = []
    for part in parts:
        warnings.extend(part["warnings"])
    return {
        "count_min": count_min,
        "count_max": count_max,
        "count_open_upper": count_open_upper,
        "warnings": sorted(set(warnings)),
    }


def extract_numeric_bounds(text: str) -> tuple[float | None, float | None, bool, str | None]:
    text = normalize_key(text)
    if not text:
        return None, None, False, None
    if text.startswith("meno di "):
        text = "<" + text.removeprefix("meno di ").strip()
    if text.startswith("piu di "):
        text = ">" + text.removeprefix("piu di ").strip()
    range_match = re.search(r"(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)", text)
    if range_match:
        return float(range_match.group(1).replace(",", ".")), float(range_match.group(2).replace(",", ".")), False, "range"
    lower_open = re.search(r">\s*(\d+(?:[.,]\d+)?)", text)
    if lower_open:
        lower = float(lower_open.group(1).replace(",", "."))
        return lower, None, True, "open_upper"
    upper_only = re.search(r"<\s*(\d+(?:[.,]\d+)?)", text)
    if upper_only:
        upper = float(upper_only.group(1).replace(",", "."))
        return 0.0, upper, False, "upper_only"
    single_value = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if single_value:
        value = float(single_value.group(1).replace(",", "."))
        return value, value, False, "single"
    return None, None, False, None


def parse_frequency_bounds(value: Any) -> dict[str, Any]:
    text = normalize_key(value)
    if not text:
        return {
            "usage_frequency_per_week_min": None,
            "usage_frequency_per_week_max": None,
            "usage_frequency_per_week_open_upper": False,
            "warnings": [],
        }
    if text == "mai":
        return {
            "usage_frequency_per_week_min": 0.0,
            "usage_frequency_per_week_max": 0.0,
            "usage_frequency_per_week_open_upper": False,
            "warnings": [],
        }
    if text == "quasi ogni giorno":
        return {
            "usage_frequency_per_week_min": 5.0,
            "usage_frequency_per_week_max": 7.0,
            "usage_frequency_per_week_open_upper": False,
            "warnings": [],
        }
    if text == "ogni giorno":
        return {
            "usage_frequency_per_week_min": 7.0,
            "usage_frequency_per_week_max": 7.0,
            "usage_frequency_per_week_open_upper": False,
            "warnings": [],
        }
    if text == "meno di una volta a settimana":
        return {
            "usage_frequency_per_week_min": 0.0,
            "usage_frequency_per_week_max": 1.0,
            "usage_frequency_per_week_open_upper": False,
            "warnings": [],
        }
    text = text.replace("una volta", "1 volta")
    lower, upper, open_upper, kind = extract_numeric_bounds(text)
    if lower is None:
        return {
            "usage_frequency_per_week_min": None,
            "usage_frequency_per_week_max": None,
            "usage_frequency_per_week_open_upper": False,
            "warnings": ["frequency_unparsed"],
        }
    multiplier = 7.0 if "al giorno" in text else 1.0
    if kind == "open_upper" and lower is not None:
        lower += 1.0
    return {
        "usage_frequency_per_week_min": lower * multiplier,
        "usage_frequency_per_week_max": None if upper is None else upper * multiplier,
        "usage_frequency_per_week_open_upper": open_upper,
        "warnings": [],
    }


def parse_duration_minutes_bounds(value: Any) -> dict[str, Any]:
    text = normalize_key(value)
    if not text:
        return {
            "duration_minutes_min": None,
            "duration_minutes_max": None,
            "duration_minutes_open_upper": False,
            "warnings": [],
        }
    lower, upper, open_upper, _ = extract_numeric_bounds(text)
    if lower is None:
        return {
            "duration_minutes_min": None,
            "duration_minutes_max": None,
            "duration_minutes_open_upper": False,
            "warnings": ["duration_unparsed"],
        }
    multiplier = 60.0 if ("ora" in text or "ore" in text or re.search(r"\d+h|\bh\b", text)) else 1.0
    return {
        "duration_minutes_min": lower * multiplier,
        "duration_minutes_max": None if upper is None else upper * multiplier,
        "duration_minutes_open_upper": open_upper,
        "warnings": [],
    }


def parse_daily_hours_bounds(value: Any) -> dict[str, Any]:
    text = normalize_key(value)
    if not text:
        return {
            "daily_usage_hours_min": None,
            "daily_usage_hours_max": None,
            "daily_usage_hours_open_upper": False,
            "warnings": [],
        }
    lower, upper, open_upper, _ = extract_numeric_bounds(text)
    if lower is None:
        return {
            "daily_usage_hours_min": None,
            "daily_usage_hours_max": None,
            "daily_usage_hours_open_upper": False,
            "warnings": ["daily_hours_unparsed"],
        }
    multiplier = 1.0
    if "minut" in text:
        multiplier = 1.0 / 60.0
    return {
        "daily_usage_hours_min": lower * multiplier,
        "daily_usage_hours_max": None if upper is None else upper * multiplier,
        "daily_usage_hours_open_upper": open_upper,
        "warnings": [],
    }


def parse_time_window(value: Any) -> dict[str, Any]:
    text = normalize_key(value)
    if not text:
        return {
            "start_window_start_min": None,
            "start_window_end_min": None,
            "start_window_wraps_next_day": False,
            "warnings": [],
        }
    match = re.search(r"(\d{1,2})\s*[-:]\s*(\d{1,2})", text)
    if match:
        start_hour = int(match.group(1))
        end_hour = int(match.group(2))
        start_min = start_hour * 60
        end_min = end_hour * 60
        return {
            "start_window_start_min": start_min,
            "start_window_end_min": end_min,
            "start_window_wraps_next_day": end_min <= start_min,
            "warnings": [],
        }
    for label, (start_min, end_min, wraps) in TIME_WINDOW_MAP.items():
        if label in text:
            return {
                "start_window_start_min": start_min,
                "start_window_end_min": end_min,
                "start_window_wraps_next_day": wraps,
                "warnings": [],
            }
    return {
        "start_window_start_min": None,
        "start_window_end_min": None,
        "start_window_wraps_next_day": False,
        "warnings": ["time_window_unparsed"],
    }


def month_range(start_month: int, end_month: int) -> list[int]:
    months = [start_month]
    current = start_month
    while current != end_month:
        current = 1 if current == 12 else current + 1
        months.append(current)
    return months


def parse_active_months(value: Any) -> tuple[list[int] | None, list[str]]:
    text = normalize_key(value)
    if not text:
        return None, []
    if "sempre" in text:
        return list(range(1, 13)), []
    if "estiv" in text or "estate" in text:
        season_months = [6, 7, 8, 9]
        if "invern" in text:
            return sorted(set(season_months + [11, 12, 1, 2, 3])), []
        return season_months, []
    if "invern" in text:
        return [11, 12, 1, 2, 3], []
    found_months = [month for name, month in MONTH_NAME_TO_NUM.items() if name in text]
    if len(found_months) >= 2:
        return month_range(found_months[0], found_months[1]), []
    if len(found_months) == 1:
        return found_months, []
    return None, ["active_months_unparsed"]


def parse_surface_band(value: Any) -> dict[str, Any]:
    text = normalize_key(value)
    if not text:
        return {"surface_sqm_min": None, "surface_sqm_max": None, "surface_sqm_open_upper": False}
    lower, upper, open_upper, _ = extract_numeric_bounds(text)
    return {
        "surface_sqm_min": lower,
        "surface_sqm_max": upper,
        "surface_sqm_open_upper": open_upper,
    }


def parse_year_band(value: Any) -> dict[str, Any]:
    text = normalize_key(value)
    if not text or text == "non so":
        return {"construction_year_min": None, "construction_year_max": None, "construction_year_open_upper": False}
    if text.startswith("prima del"):
        lower, upper, _, _ = extract_numeric_bounds(text)
        return {"construction_year_min": None, "construction_year_max": None if upper is None else upper - 1, "construction_year_open_upper": False}
    lower, upper, open_upper, _ = extract_numeric_bounds(text)
    return {
        "construction_year_min": lower,
        "construction_year_max": upper,
        "construction_year_open_upper": open_upper,
    }


def code_from_mapping(value: Any, mapping: dict[str, int]) -> int | None:
    return mapping.get(normalize_key(value))


def parse_weekday_presence_summary(values: list[Any]) -> dict[str, int | None]:
    morning = 0
    afternoon = 0
    evening = 0
    reported = 0
    for value in values:
        text = normalize_key(value)
        if not text:
            continue
        reported += 1
        for score_label, score_value in PRESENCE_SCORE.items():
            if f"mattina - {score_label}" in text or f"mattina - {score_label}".replace(" ", "  ") in text:
                morning += score_value
            if f"pomeriggio - {score_label}" in text:
                afternoon += score_value
            if f"sera - {score_label}" in text:
                evening += score_value
    return {
        "weekday_people_reported": reported,
        "weekday_morning_presence_score_total": morning,
        "weekday_afternoon_presence_score_total": afternoon,
        "weekday_evening_presence_score_total": evening,
    }


def build_column_map(df: pd.DataFrame) -> dict[str, Any]:
    locator = ColumnLocator([str(column) for column in df.columns])
    person_columns = locator.find_all("riempire la tabella con la presenza media in casa nei giorni feriali")
    return {
        "timestamp": locator.find("informazioni cronologiche"),
        "imei": locator.find("codice imei dispositivo"),
        "home_type": locator.find("tipologia di abitazione"),
        "surface": locator.find("superficie indicativa"),
        "construction_year": locator.find("anno di costruzione"),
        "room_count": locator.find("numero di stanze"),
        "heating_type": locator.find("tipo di riscaldamento principale"),
        "has_pv": locator.find("presenza di pannelli fotovoltaici"),
        "resident_count": locator.find("numero di persone che vivono stabilmente in casa"),
        "weekend_pattern": locator.find("nei weekend siete generalmente"),
        "recent_change": locator.find("acquisto o dismissione di grandi elettrodomestici"),
        "long_absence": locator.find("periodi prolungati di assenza"),
        "energy_cost_change": locator.find("cambiamenti di abitudini dovuti al costo dell'energia"),
        "free_comment": locator.find("spazio per commenti"),
        "person_columns": person_columns,
        "presence": {
            output_name: locator.find("riempire la seguente tabella su presenza", label)
            for output_name, label in PRESENCE_COLUMN_LABELS.items()
        },
        "detail": {
            key: locator.find(*patterns)
            for key, patterns in DETAIL_COLUMN_PATTERNS.items()
        },
    }


def apply_bounds(payload: dict[str, Any], updates: dict[str, Any]) -> None:
    warnings = payload["warnings"]
    for key, value in updates.items():
        if key == "warnings":
            warnings.extend(value)
        else:
            payload[key] = value
    payload["warnings"] = sorted(set(warnings))


def infer_presence_from_usage(payload: dict[str, Any]) -> None:
    detail_present = any(
        payload[key] is not None
        for key in [
            "usage_frequency_per_week_min",
            "duration_minutes_min",
            "start_window_start_min",
            "daily_usage_hours_min",
            "active_months",
        ]
    )
    if detail_present and not payload["present"]:
        payload["present"] = True
        payload["count_min"] = 1
        payload["count_max"] = 1
        payload["warnings"] = sorted(set(payload["warnings"] + ["presence_inferred_from_usage_fields"]))


def build_device_payloads(row: pd.Series, column_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    devices = {device_name: empty_device_payload() for device_name in OUTPUT_DEVICE_ORDER}
    presence_parts: dict[str, dict[str, Any]] = {}

    for device_name, column_name in column_map["presence"].items():
        parsed = parse_count_bounds(row[column_name] if column_name else None)
        presence_parts[device_name] = parsed

    primary_fridge = presence_parts["Frigorifero principale"]
    secondary_fridge = presence_parts["Frigorifero secondario"]
    electric_hob = presence_parts["Piano cottura elettrico"]
    induction_hob = presence_parts["Piano cottura induzione"]
    hob_total = combine_count_bounds(electric_hob, induction_hob)

    direct_device_map = {
        "Frigorifero principale": primary_fridge,
        "Frigorifero secondario": secondary_fridge,
        "Congelatore": presence_parts["Congelatore"],
        "Lavatrice": presence_parts["Lavatrice"],
        "Asciugatrice": presence_parts["Asciugatrice"],
        "Lavastoviglie": presence_parts["Lavastoviglie"],
        "Forno": presence_parts["Forno"],
        "Piano cottura": hob_total,
        "Microonde": presence_parts["Microonde"],
        "Boiler": presence_parts["Boiler"],
        "Climatizzatore": presence_parts["Climatizzatore"],
        "Televisori": presence_parts["Televisori"],
        "Computer": presence_parts["Computer"],
        "Console": presence_parts["Console"],
        "Auto elettrica": presence_parts["Auto elettrica"],
    }

    for device_name, count_info in direct_device_map.items():
        payload = devices[device_name]
        payload["count_min"] = count_info["count_min"]
        payload["count_max"] = count_info["count_max"]
        payload["count_open_upper"] = count_info["count_open_upper"]
        payload["present"] = count_info["count_min"] > 0 or count_info["count_open_upper"]
        payload["warnings"] = sorted(set(count_info["warnings"]))

    detail_map = column_map["detail"]
    apply_bounds(devices["Lavatrice"], parse_frequency_bounds(row[detail_map["lavatrice_frequency"]] if detail_map["lavatrice_frequency"] else None))
    apply_bounds(devices["Lavatrice"], parse_time_window(row[detail_map["lavatrice_time"]] if detail_map["lavatrice_time"] else None))
    apply_bounds(devices["Lavatrice"], parse_duration_minutes_bounds(row[detail_map["lavatrice_duration"]] if detail_map["lavatrice_duration"] else None))

    apply_bounds(devices["Lavastoviglie"], parse_frequency_bounds(row[detail_map["lavastoviglie_frequency"]] if detail_map["lavastoviglie_frequency"] else None))
    apply_bounds(devices["Lavastoviglie"], parse_time_window(row[detail_map["lavastoviglie_time"]] if detail_map["lavastoviglie_time"] else None))
    apply_bounds(devices["Lavastoviglie"], parse_duration_minutes_bounds(row[detail_map["lavastoviglie_duration"]] if detail_map["lavastoviglie_duration"] else None))

    apply_bounds(devices["Forno"], parse_frequency_bounds(row[detail_map["forno_frequency"]] if detail_map["forno_frequency"] else None))
    apply_bounds(devices["Forno"], parse_duration_minutes_bounds(row[detail_map["forno_duration"]] if detail_map["forno_duration"] else None))

    apply_bounds(devices["Climatizzatore"], parse_daily_hours_bounds(row[detail_map["climate_daily_hours"]] if detail_map["climate_daily_hours"] else None))
    active_months, month_warnings = parse_active_months(row[detail_map["climate_period"]] if detail_map["climate_period"] else None)
    devices["Climatizzatore"]["active_months"] = active_months
    devices["Climatizzatore"]["warnings"] = sorted(set(devices["Climatizzatore"]["warnings"] + month_warnings))

    free_comment = normalize_key(row[column_map["free_comment"]] if column_map["free_comment"] else None)
    if free_comment and "forno" in free_comment and "microonde" in free_comment and "rifer" in free_comment:
        devices["Forno"]["warnings"] = sorted(set(devices["Forno"]["warnings"] + ["comment_mentions_forno_microonde_correction"]))
        devices["Microonde"]["warnings"] = sorted(set(devices["Microonde"]["warnings"] + ["comment_mentions_forno_microonde_correction"]))

    for payload in devices.values():
        infer_presence_from_usage(payload)

    return devices


def build_household_record(row: pd.Series, column_map: dict[str, Any], imei: str) -> dict[str, Any]:
    presence_summary = parse_weekday_presence_summary([row[column] for column in column_map["person_columns"]])
    comments_present = parse_reported_flag(row[column_map["free_comment"]] if column_map["free_comment"] else None)
    record = {
        "imei": imei,
        "response_timestamp": clean_text(row[column_map["timestamp"]] if column_map["timestamp"] else None),
        "stable_residents_count": parse_int_like(row[column_map["resident_count"]] if column_map["resident_count"] else None),
        "room_count": parse_int_like(row[column_map["room_count"]] if column_map["room_count"] else None),
        "home_type_code": code_from_mapping(row[column_map["home_type"]] if column_map["home_type"] else None, HOME_TYPE_CODE),
        "heating_type_code": code_from_mapping(row[column_map["heating_type"]] if column_map["heating_type"] else None, HEATING_TYPE_CODE),
        "has_pv": parse_binary_yes_no(row[column_map["has_pv"]] if column_map["has_pv"] else None),
        "weekend_pattern_code": code_from_mapping(row[column_map["weekend_pattern"]] if column_map["weekend_pattern"] else None, WEEKEND_PATTERN_CODE),
        "recent_large_appliance_change_reported": parse_reported_flag(row[column_map["recent_change"]] if column_map["recent_change"] else None),
        "long_absence_reported": parse_reported_flag(row[column_map["long_absence"]] if column_map["long_absence"] else None),
        "energy_cost_habit_change_reported": parse_reported_flag(row[column_map["energy_cost_change"]] if column_map["energy_cost_change"] else None),
        "free_comment_reported": comments_present,
    }
    record.update(parse_surface_band(row[column_map["surface"]] if column_map["surface"] else None))
    record.update(parse_year_band(row[column_map["construction_year"]] if column_map["construction_year"] else None))
    record.update(presence_summary)
    return record


def load_input(path: Path, sheet_name: str | int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def flatten_device_rows(imei: str, devices: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for device_name in OUTPUT_DEVICE_ORDER:
        payload = devices[device_name]
        row = {"imei": imei, "device": device_name}
        for key, value in payload.items():
            if key == "warnings":
                row["warning_codes"] = "|".join(value)
            elif key == "active_months":
                row["active_months"] = "|".join(str(month) for month in value) if value else ""
            else:
                row[key] = value
        rows.append(row)
    return rows


def write_json_outputs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = output_dir / f"{record['imei']}.json"
        json_record = {
            **record,
            "devices": {
                device_name: {
                    key: value
                    for key, value in payload.items()
                    if key != "warnings"
                }
                for device_name, payload in record["devices"].items()
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(json_record, handle, ensure_ascii=False, indent=2)


def build_outputs(df: pd.DataFrame, column_map: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    household_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    json_records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        imei = normalize_imei(row[column_map["imei"]] if column_map["imei"] else None)
        if not imei:
            continue
        devices = build_device_payloads(row, column_map)
        household_rows.append(build_household_record(row, column_map, imei))
        device_rows.extend(flatten_device_rows(imei, devices))
        json_records.append(
            {
                "schema_version": 2,
                "imei": imei,
                "devices": {device_name: devices[device_name] for device_name in OUTPUT_DEVICE_ORDER},
            }
        )
    return household_rows, device_rows, json_records


def write_excel_output(path: Path, household_rows: list[dict[str, Any]], device_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    household_df = pd.DataFrame(household_rows).sort_values("imei")
    device_df = pd.DataFrame(device_rows).sort_values(["imei", "device"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        household_df.to_excel(writer, sheet_name="households", index=False)
        device_df.to_excel(writer, sheet_name="device_usage", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact Excel and numeric per-IMEI JSON outputs from questionnaire exports.",
    )
    parser.add_argument("--input", required=True, help="Path to raw questionnaire export (.csv or .xlsx)")
    parser.add_argument("--sheet-name", default=0, help="Excel sheet name or index when reading .xlsx inputs")
    parser.add_argument("--output-excel", default=str(DEFAULT_OUTPUT_EXCEL), help="Target Excel workbook path")
    parser.add_argument("--output-json-dir", default=str(DEFAULT_OUTPUT_JSON_DIR), help="Target directory for normalized per-IMEI JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    sheet_name: str | int = int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    df = load_input(input_path, sheet_name)
    column_map = build_column_map(df)
    household_rows, device_rows, json_records = build_outputs(df, column_map)

    write_excel_output(Path(args.output_excel), household_rows, device_rows)
    write_json_outputs(Path(args.output_json_dir), json_records)

    print(f"Processed {len(json_records)} IMEI records from {input_path}")
    print(f"Excel output: {Path(args.output_excel)}")
    print(f"JSON output directory: {Path(args.output_json_dir)}")


if __name__ == "__main__":
    main()