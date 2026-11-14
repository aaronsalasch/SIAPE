"""API de predicción para el dashboard y el reporte.

Uso:
    from src.ml.predecir import predecir_consumo, pronostico_24h
    predecir_consumo("RES-001", "2026-06-20 20:00:00", 31.5, 75)
    pronostico_24h("COM-001", "2026-06-20", temperaturas=None)

Cómo se obtienen las variables históricas (lags)
------------------------------------------------
El modelo usa el consumo de 1 h, 24 h y 168 h antes y la media de las 24 h previas. Al
predecir una fecha hay tres situaciones, en este orden de prioridad:
  1. La hora necesaria existe en el histórico procesado -> se usa el valor real.
  2. Ya se predijo en esta misma llamada (``pronostico_24h`` predice hora por hora y
     reutiliza sus propias predicciones como lags) -> se usa la predicción.
  3. No existe (fecha futura) -> se usa el último valor histórico de la MISMA hora del
     día (y del mismo día de la semana para el lag de 168 h) de esa instalación; si no
     hay, el promedio de la instalación a esa hora.
El intervalo es el rango entre los percentiles 5 y 95 de las predicciones de los árboles
del Random Forest (si el modelo no es RF, predicción ± RMSE de prueba).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.data.generar_datos import FESTIVOS_2026  # noqa: E402
from src.data.preparar_datos import agregar_variables_calendario  # noqa: E402

_INSTALACIONES = {i["id"]: i for i in config.INSTALACIONES}


# --------------------------------------------------------------------------------------
# Carga con caché
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cargar_modelo():
    """Pipeline entrenado y sus métricas (se cargan una sola vez)."""
    import json
    modelo = joblib.load(config.MODELO_CONSUMO)
    metricas = json.loads(config.METRICAS_ML.read_text(encoding="utf-8"))
    return modelo, metricas


@lru_cache(maxsize=1)
def _cargar_historico() -> dict:
    """Series de consumo, temperatura y humedad por instalación y tablas de respaldo."""
    df = pd.read_csv(config.CSV_CONSUMO_PROC, usecols=["id_instalacion", "fecha_hora", "temperatura_c", "humedad_pct", "consumo_kwh"])
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["hora"] = df["fecha_hora"].dt.hour
    hist = {}
    for id_inst, g in df.groupby("id_instalacion"):
        g = g.sort_values("fecha_hora").set_index("fecha_hora")
        hist[id_inst] = {
            "consumo": g["consumo_kwh"],
            "media_hora": g.groupby("hora")["consumo_kwh"].mean(),
            "temp_hora": g.tail(24 * 14).groupby("hora")["temperatura_c"].mean(),      # últimas 2 semanas
            "hum_hora": g.tail(24 * 14).groupby("hora")["humedad_pct"].mean(),
            "ultimo": g.index.max(),
            "media": float(g["consumo_kwh"].mean()),
        }
    return hist


def _validar_id(id_instalacion: str) -> str:
    id_instalacion = str(id_instalacion).strip().upper()
    if id_instalacion not in _INSTALACIONES:
        raise ValueError(f"Instalación desconocida: {id_instalacion!r}. Válidas: {', '.join(_INSTALACIONES)}")
    return id_instalacion


# --------------------------------------------------------------------------------------
# Lags
# --------------------------------------------------------------------------------------
def _valor_pasado(h: dict, ts: pd.Timestamp, horas_atras: int, extra: dict) -> float:
    """Consumo en ``ts - horas_atras`` (histórico > predicciones previas > respaldo)."""
    t = ts - pd.Timedelta(hours=horas_atras)
    serie = h["consumo"]
    if t in serie.index:
        return float(serie.loc[t])
    if t in extra:
        return float(extra[t])
    # Respaldo: último valor histórico de la misma hora del día (y mismo día de la semana si 168 h)
    paso = 7 if horas_atras == 168 else 1
    candidato = serie.index[(serie.index.hour == t.hour) & (serie.index < ts)]
    if paso == 7:
        candidato = candidato[candidato.dayofweek == t.dayofweek]
    if len(candidato):
        return float(serie.loc[candidato.max()])
    return float(h["media_hora"].get(t.hour, h["media"]))


def _media_movil_24h(h: dict, ts: pd.Timestamp, extra: dict) -> float:
    """Media de las 24 horas previas a ``ts`` (con los mismos criterios de prioridad)."""
    return float(np.mean([_valor_pasado(h, ts, k, extra) for k in range(1, 25)]))


# --------------------------------------------------------------------------------------
# Construcción de entradas y predicción
# --------------------------------------------------------------------------------------
def _construir_fila(id_inst: str, ts: pd.Timestamp, temp: float, hum: float, festivo: int, extra: dict) -> pd.DataFrame:
    inst, h = _INSTALACIONES[id_inst], _cargar_historico()[id_inst]
    fila = pd.DataFrame([{
        "fecha_hora": ts, "es_festivo": int(festivo), "temperatura_c": float(temp), "humedad_pct": float(hum),
        "tipo_instalacion": inst["tipo"], "ciudad": inst["ciudad"],
        "superficie_m2": inst["superficie_m2"], "num_ocupantes": inst["ocupantes"],
        "consumo_lag_1h": _valor_pasado(h, ts, 1, extra),
        "consumo_lag_24h": _valor_pasado(h, ts, 24, extra),
        "consumo_lag_168h": _valor_pasado(h, ts, 168, extra),
        "consumo_media_movil_24h": _media_movil_24h(h, ts, extra),
    }])
    return agregar_variables_calendario(fila)       # mismas definiciones que en el entrenamiento


def _predecir_fila(fila: pd.DataFrame) -> tuple[float, float, float, str]:
    modelo, metricas = _cargar_modelo()
    X = fila[list(modelo.feature_names_in_)]
    pred = float(modelo.predict(X)[0])
    nombre = metricas["seleccionado"]
    ultimo = modelo.steps[-1][1]
    if hasattr(ultimo, "estimators_"):                      # Random Forest: percentiles de los árboles
        Xt = modelo[:-1].transform(X)
        por_arbol = np.array([t.predict(Xt)[0] for t in ultimo.estimators_])
        lo, hi = np.percentile(por_arbol, [5, 95])
    else:                                                   # otro modelo: ± RMSE de prueba
        rmse = metricas["modelos"][nombre]["rmse"]
        lo, hi = pred - rmse, pred + rmse
    pred = max(pred, 0.0)
    return pred, max(min(lo, pred), 0.0), max(hi, pred), nombre


def _resultado(id_inst, ts, fila, pred, lo, hi, nombre) -> dict:
    enteras = {"es_festivo", "superficie_m2", "num_ocupantes", "hora", "dia_semana_num", "mes", "dia_anio", "es_fin_semana"}
    entradas = {}
    for c, v in fila.drop(columns="fecha_hora").iloc[0].items():
        if c in enteras:
            entradas[c] = int(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            entradas[c] = round(float(v), 4)
        else:
            entradas[c] = v
    return {"id_instalacion": id_inst, "fecha_hora": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "prediccion_kwh": round(float(pred), 3), "intervalo": [round(float(lo), 3), round(float(hi), 3)],
            "modelo": nombre, "entradas": entradas}


def predecir_consumo(id_instalacion: str, fecha_hora: str, temperatura_c: float, humedad_pct: float,
                     es_festivo: int = 0) -> dict:
    """Predice el consumo (kWh) de una instalación en una hora concreta.

    Devuelve {"id_instalacion","fecha_hora","prediccion_kwh","intervalo":[lo, hi],"modelo","entradas":{...}}.
    """
    id_inst = _validar_id(id_instalacion)
    ts = pd.Timestamp(fecha_hora).floor("h")
    fila = _construir_fila(id_inst, ts, temperatura_c, humedad_pct, es_festivo, {})
    pred, lo, hi, nombre = _predecir_fila(fila)
    return _resultado(id_inst, ts, fila, pred, lo, hi, nombre)


def pronostico_24h(id_instalacion: str, fecha: str, temperaturas: list | None = None) -> list[dict]:
    """Pronóstico hora por hora (00:00 a 23:00) de un día.

    ``temperaturas``: 24 valores en °C o None (se usa la temperatura típica por hora de las
    últimas dos semanas del histórico). Cada predicción se reutiliza como lag de la siguiente.
    """
    id_inst = _validar_id(id_instalacion)
    h = _cargar_historico()[id_inst]
    dia = pd.Timestamp(fecha).normalize()
    if temperaturas is not None and len(temperaturas) != 24:
        raise ValueError("temperaturas debe tener 24 valores (una por hora) o ser None")
    festivo = int(dia.strftime("%Y-%m-%d") in FESTIVOS_2026)
    extra: dict = {}
    salida = []
    for hora in range(24):
        ts = dia + pd.Timedelta(hours=hora)
        temp = float(temperaturas[hora]) if temperaturas is not None else float(h["temp_hora"].get(hora, 25.0))
        hum = float(h["hum_hora"].get(hora, 75.0))
        fila = _construir_fila(id_inst, ts, temp, hum, festivo, extra)
        pred, lo, hi, nombre = _predecir_fila(fila)
        extra[ts] = pred
        salida.append(_resultado(id_inst, ts, fila, pred, lo, hi, nombre))
    return salida


if __name__ == "__main__":
    print(predecir_consumo("RES-001", "2026-06-20 20:00:00", 31.5, 75))
    for r in pronostico_24h("COM-001", "2026-07-02")[:3]:
        print(r["fecha_hora"], r["prediccion_kwh"], r["intervalo"])
