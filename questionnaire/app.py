"""Prototipo Streamlit del nuovo questionario NILM (schema v3).

Avvio:  streamlit run questionnaire/app.py
"""

import datetime as dt
import io
import os
import uuid

import pandas as pd
import streamlit as st

from schema import (
    CONFIDENCE_LEVELS,
    EXCEL_COLUMNS,
    MONTHS,
    DeviceInstance,
    Household,
    catalog_by_type,
    load_catalog,
    lookup_model,
    minutes_to_hhmm,
    to_json,
    to_rows,
    validate,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

NO_MODEL = "— non specificato —"

POWER_SOURCE_HINT = {
    "default": "Valori tipici per questa tipologia: seleziona un modello qui sopra "
               "o modificali a mano per rendere la stima piu' precisa.",
    "catalogo": "Valori precompilati dal modello selezionato. Puoi comunque modificarli.",
    "dichiarato": "Valori dichiarati da te.",
}

st.set_page_config(page_title="Questionario NILM", page_icon="⚡", layout="wide")


@st.cache_data
def get_catalog() -> dict:
    return load_catalog()


CATALOG = get_catalog()
BY_TYPE = catalog_by_type(CATALOG)
TYPE_NAMES = list(BY_TYPE.keys())

if "devices" not in st.session_state:
    st.session_state.devices = []


def make_instance(device_type: str) -> DeviceInstance:
    """Create a device instance prefilled with the catalog defaults for its type."""
    entry = BY_TYPE[device_type]
    power = entry.get("power", {})
    duration = entry.get("duration_minutes", {})
    mode = entry.get("usage_mode", "cycle")

    instance = DeviceInstance(
        id=f"{device_type.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}",
        type=device_type,
        usage_mode=mode,
        power_source="default",
        p_min_w=power.get("min_w"),
        p_typical_w=power.get("typical_w"),
        p_max_w=power.get("max_w"),
    )
    if mode == "cycle":
        instance.duration_minutes_min = duration.get("min")
        instance.duration_minutes_max = duration.get("typical")
    return instance


def seed_widget(widget_key: str, value) -> None:
    """Seed a widget's default once.

    Streamlit ignores a widget's `value=`/`default=` argument on every rerun
    after the first: once the widget has a `key`, session_state is the source of
    truth. So defaults are seeded here, and any programmatic overwrite (picking a
    model) must write to session_state too, not just to the DeviceInstance.
    """
    if widget_key not in st.session_state:
        st.session_state[widget_key] = value


def mark_declared(device: DeviceInstance) -> None:
    """Callback: the user edited a power field by hand."""
    device.power_source = "dichiarato"


def apply_model(device: DeviceInstance) -> None:
    """Callback: a catalog model was picked (or cleared) — push its powers into the widgets."""
    picked = st.session_state[f"model-{device.id}"]

    if picked == NO_MODEL:
        entry = BY_TYPE[device.type]
        power = entry.get("power", {})
        device.brand = device.model = None
        device.power_source = "default"
        st.session_state[f"ptyp-{device.id}"] = float(power.get("typical_w") or 0.0)
        st.session_state[f"pmin-{device.id}"] = float(power.get("min_w") or 0.0)
        st.session_state[f"pmax-{device.id}"] = float(power.get("max_w") or 0.0)
        return

    brand, model_name = picked.split(" · ", 1)
    found = lookup_model(CATALOG, device.type, brand, model_name)
    if found is None:
        return

    device.brand, device.model = brand, model_name
    device.power_source = "catalogo"
    st.session_state[f"ptyp-{device.id}"] = float(found.get("typical_w") or 0.0)
    st.session_state[f"pmin-{device.id}"] = float(found.get("min_w") or 0.0)
    st.session_state[f"pmax-{device.id}"] = float(found.get("max_w") or 0.0)

    duration = found.get("duration_min")
    if duration and device.usage_mode == "cycle":
        current = st.session_state.get(f"dur-{device.id}", (0, int(duration)))
        st.session_state[f"dur-{device.id}"] = (current[0], int(duration))


def time_to_minutes(value: dt.time) -> int:
    return value.hour * 60 + value.minute


def minutes_to_time(minutes) -> dt.time:
    minutes = int(minutes or 0) % (24 * 60)
    return dt.time(minutes // 60, minutes % 60)


# ---------------------------------------------------------------- sidebar ----

with st.sidebar:
    st.header("Dati abitazione")
    imei = st.text_input("Codice IMEI dispositivo", key="imei", help="Solo cifre.")
    contract_power_kw = st.number_input(
        "Potenza contrattuale (kW)",
        min_value=0.0, max_value=100.0, value=3.0, step=0.5,
        help="Il valore sul contratto di fornitura (tipicamente 3.0 o 4.5 kW).",
    )
    st.divider()
    st.subheader("Carico base")
    base_load_w = st.number_input(
        "Consumo sempre presente (W)",
        min_value=0.0, max_value=5000.0, value=50.0, step=10.0,
        help="Router, allarme, decoder, standby vari: cio' che non si spegne mai. "
             "Aiuta il modello a non attribuire questa quota agli elettrodomestici.",
    )
    base_load_confidence = st.select_slider(
        "Quanto e' affidabile questa stima?", options=CONFIDENCE_LEVELS, value="bassa"
    )
    st.divider()
    st.caption(
        "Non vengono piu' richiesti: tipologia abitazione, superficie, anno di "
        "costruzione, numero stanze, riscaldamento, pannelli, numero di persone, "
        "fasce d'eta', presenza in casa e note libere."
    )

# ------------------------------------------------------------------- main ----

st.title("⚡ Questionario consumi domestici")
st.markdown(
    "Aggiungi un dispositivo alla volta. Per ognuno indica **quanto consuma** e "
    "**quando lo usi**: piu' campi compili, piu' la stima dei consumi sara' precisa. "
    "I campi lasciati vuoti non sono un problema, il modello usa i valori tipici."
)

col_add, col_spacer = st.columns([2, 3])
with col_add:
    chosen_type = st.selectbox("Tipologia dispositivo", TYPE_NAMES, key="new_device_type")
    if st.button("➕ Aggiungi dispositivo", type="primary", width='stretch'):
        st.session_state.devices.append(make_instance(chosen_type))
        st.rerun()

st.divider()

if not st.session_state.devices:
    st.info("Nessun dispositivo inserito. Usa **Aggiungi dispositivo** per iniziare.")

remove_index = None

for index, device in enumerate(st.session_state.devices):
    entry = BY_TYPE[device.type]
    heading = f"{device.type}" + (f" — {device.label}" if device.label else "")
    if device.count > 1:
        heading += f"  ×{device.count}"

    with st.expander(heading, expanded=len(st.session_state.devices) <= 2):
        key = device.id

        head_left, head_mid, head_right = st.columns([3, 1, 1])
        device.label = head_left.text_input(
            "Etichetta (dove si trova / a chi serve)",
            value=device.label, key=f"label-{key}",
            placeholder="es. camera da letto",
            help="Serve a distinguere due dispositivi dello stesso tipo.",
        )
        device.count = head_mid.number_input(
            "Quantita'", min_value=1, max_value=20, value=device.count, step=1, key=f"count-{key}"
        )
        head_right.write("")
        if head_right.button("🗑 Rimuovi", key=f"remove-{key}", width='stretch'):
            remove_index = index

        # ----- modello e consumi -----
        st.markdown("**Consumi**")
        models = entry.get("models", [])
        model_labels = [NO_MODEL] + [f"{m['brand']} · {m['model']}" for m in models]

        seed_widget(f"ptyp-{key}", float(device.p_typical_w or 0.0))
        seed_widget(f"pmin-{key}", float(device.p_min_w or 0.0))
        seed_widget(f"pmax-{key}", float(device.p_max_w or 0.0))

        st.selectbox(
            "Marca e modello (opzionale)",
            model_labels,
            key=f"model-{key}",
            on_change=apply_model,
            args=(device,),
            help="Selezionando un modello i consumi vengono precompilati, e restano modificabili.",
        )

        p_typ_col, p_min_col, p_max_col = st.columns(3)
        p_typ_col.number_input(
            "Potenza tipica (W)", min_value=0.0, max_value=20000.0, step=10.0,
            key=f"ptyp-{key}", on_change=mark_declared, args=(device,),
        )
        p_min_col.number_input(
            "Potenza minima (W)", min_value=0.0, max_value=20000.0, step=10.0,
            key=f"pmin-{key}", on_change=mark_declared, args=(device,),
            help="Opzionale: il consumo nelle fasi piu' leggere del ciclo.",
        )
        p_max_col.number_input(
            "Potenza di picco (W)", min_value=0.0, max_value=20000.0, step=10.0,
            key=f"pmax-{key}", on_change=mark_declared, args=(device,),
            help="Opzionale: il picco assorbito, es. resistenza in riscaldamento.",
        )
        device.p_typical_w = st.session_state[f"ptyp-{key}"] or None
        device.p_min_w = st.session_state[f"pmin-{key}"] or None
        device.p_max_w = st.session_state[f"pmax-{key}"] or None
        st.caption(POWER_SOURCE_HINT[device.power_source])

        # ----- utilizzo -----
        if device.usage_mode == "always_on":
            st.markdown("**Utilizzo**")
            st.caption(
                "Dispositivo sempre alimentato: non servono orari di utilizzo, "
                "il modello lo tratta come carico continuo con cicli di compressore."
            )
        else:
            st.markdown("**Utilizzo**")

            if device.usage_mode == "cycle":
                seed_widget(f"freq-{key}", (
                    float(device.usage_frequency_per_week_min or 0.0),
                    float(device.usage_frequency_per_week_max or 3.0),
                ))
                seed_widget(f"dur-{key}", (
                    int(device.duration_minutes_min or 0),
                    int(device.duration_minutes_max or 60),
                ))

                freq_col, dur_col = st.columns(2)
                freq_min, freq_max = freq_col.slider(
                    "Quante volte a settimana?", 0.0, 21.0,
                    step=0.5, key=f"freq-{key}",
                    help="Indica un intervallo: es. da 3 a 4 volte.",
                )
                device.usage_frequency_per_week_min = freq_min
                device.usage_frequency_per_week_max = freq_max

                dur_min, dur_max = dur_col.slider(
                    "Quanto dura un ciclo? (minuti)", 0, 720,
                    step=5, key=f"dur-{key}",
                )
                device.duration_minutes_min = float(dur_min)
                device.duration_minutes_max = float(dur_max)
                window_label = "In che fascia oraria viene avviato di solito?"
            else:
                seed_widget(f"hours-{key}", (
                    float(device.daily_usage_hours_min or 0.0),
                    float(device.daily_usage_hours_max or 2.0),
                ))
                hours_min, hours_max = st.slider(
                    "Quante ore al giorno viene usato?", 0.0, 24.0,
                    step=0.5, key=f"hours-{key}",
                )
                device.daily_usage_hours_min = hours_min
                device.daily_usage_hours_max = hours_max
                window_label = "In che fascia oraria viene usato di solito?"

            st.markdown(f"*{window_label}*")
            use_window = st.checkbox(
                "Ha una fascia oraria abbastanza regolare",
                value=device.start_window_start_min is not None,
                key=f"usewin-{key}",
            )
            if use_window:
                from_col, to_col = st.columns(2)
                start = from_col.time_input(
                    "Da", value=minutes_to_time(device.start_window_start_min or 600),
                    step=1800, key=f"winstart-{key}",
                )
                end = to_col.time_input(
                    "A", value=minutes_to_time(device.start_window_end_min or 1020),
                    step=1800, key=f"winend-{key}",
                )
                device.start_window_start_min = time_to_minutes(start)
                device.start_window_end_min = time_to_minutes(end)
                device.start_window_wraps_next_day = (
                    device.start_window_end_min <= device.start_window_start_min
                )
                if device.start_window_wraps_next_day:
                    st.caption(
                        f"Fascia a cavallo della mezzanotte: "
                        f"{minutes_to_hhmm(device.start_window_start_min)} → "
                        f"{minutes_to_hhmm(device.start_window_end_min)} del giorno dopo."
                    )
            else:
                device.start_window_start_min = None
                device.start_window_end_min = None
                device.start_window_wraps_next_day = False

        # ----- stagionalita' e confidenza -----
        if entry.get("seasonal"):
            seasonal = st.checkbox(
                "Si usa solo in certi mesi dell'anno",
                value=bool(device.active_months), key=f"seasonal-{key}",
            )
            if seasonal:
                picked_months = st.multiselect(
                    "Mesi di utilizzo", MONTHS,
                    default=[MONTHS[m - 1] for m in (device.active_months or [])],
                    key=f"months-{key}",
                )
                device.active_months = [MONTHS.index(m) + 1 for m in picked_months] or None
            else:
                device.active_months = None

        device.confidence = st.select_slider(
            "Quanto sei sicuro delle risposte su questo dispositivo?",
            options=CONFIDENCE_LEVELS,
            value=device.confidence,
            key=f"conf-{key}",
            help="Le risposte a bassa confidenza vengono trattate dal modello come "
                 "vincoli morbidi, non come certezze.",
        )

if remove_index is not None:
    st.session_state.devices.pop(remove_index)
    st.rerun()

# ------------------------------------------------------------------ export ---

st.divider()
st.header("Riepilogo ed esportazione")

household = Household(
    imei=imei,
    contract_power_kw=contract_power_kw,
    base_load_w=base_load_w,
    base_load_confidence=base_load_confidence,
    devices=st.session_state.devices,
)

problems = validate(household)
if problems:
    for problem in problems:
        st.warning(problem)
else:
    st.success(f"Risposta completa: {len(household.devices)} dispositivi dichiarati.")

rows = to_rows(household)
if rows:
    st.dataframe(pd.DataFrame(rows, columns=EXCEL_COLUMNS), width='stretch', hide_index=True)

    declared_typical_w = sum((row["p_typical_w"] or 0) * row["count"] for row in rows)
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Potenza tipica dichiarata (somma)", f"{declared_typical_w / 1000:.1f} kW")
    metric_right.metric("Potenza contrattuale", f"{contract_power_kw:.1f} kW")
    if declared_typical_w / 1000 > contract_power_kw:
        st.caption(
            "La somma supera la potenza contrattuale: e' normale, i dispositivi non "
            "vanno tutti insieme. Il modello sfrutta proprio questo vincolo."
        )

json_payload = to_json(household)
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    pd.DataFrame(rows, columns=EXCEL_COLUMNS).to_excel(writer, index=False, sheet_name="devices")
    pd.DataFrame([{
        "imei": household.imei,
        "contract_power_kw": household.contract_power_kw,
        "base_load_w": household.base_load_w,
        "base_load_confidence": household.base_load_confidence,
        "schema_version": 3,
    }]).to_excel(writer, index=False, sheet_name="household")

file_stem = (imei.strip() or "questionario")

download_json, download_excel, save_local = st.columns(3)
download_json.download_button(
    "⬇ Scarica JSON", json_payload, file_name=f"{file_stem}.json",
    mime="application/json", width='stretch', disabled=bool(problems),
)
download_excel.download_button(
    "⬇ Scarica Excel", excel_buffer.getvalue(), file_name=f"{file_stem}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width='stretch', disabled=bool(problems),
)
if save_local.button("💾 Salva in questionnaire/output/", width='stretch', disabled=bool(problems)):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, f"{file_stem}.json")
    excel_path = os.path.join(OUTPUT_DIR, f"{file_stem}.xlsx")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_payload)
    with open(excel_path, "wb") as f:
        f.write(excel_buffer.getvalue())
    st.success(f"Salvati:\n\n- `{json_path}`\n- `{excel_path}`")

with st.expander("Anteprima JSON (schema v3)"):
    st.code(json_payload, language="json")
