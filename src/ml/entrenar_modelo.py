"""Aprendizaje SUPERVISADO: predicción del consumo eléctrico horario (kWh).

Ejecución:  python -m src.ml.entrenar_modelo

Tipo de problema : regresión supervisada (objetivo ``consumo_kwh``).
Entrada          : ``config.CSV_CONSUMO_PROC`` (ya limpio, con la columna ``conjunto``).
Salidas          : modelo (Pipeline completo) en ``config.MODELO_CONSUMO``, métricas en
                   ``config.METRICAS_ML``, ``outputs/predicciones_test.csv``, figuras en
                   ``config.DIR_FIGURAS`` y evidencia en ``entrenamiento_modelo.txt``.

Método
------
1. Se usa la división temporal ya hecha en la preparación (entrenamiento = primeras
   ~80 % de las horas; prueba = último 20 %). No se baraja nada.
2. Cada modelo es un ``Pipeline``: ``ColumnTransformer`` (StandardScaler para las numéricas,
   OneHotEncoder para ``tipo_instalacion`` y ``ciudad``) + el regresor. Así el escalado se
   ajusta solo con los datos de entrenamiento.
3. Se comparan: Regresión Lineal, Árbol de Decisión, KNN, Random Forest y SVR (este último
   sobre una muestra, por costo). Árbol, KNN y RF ajustan hiperparámetros con una búsqueda
   pequeña (GridSearchCV) y validación cruzada TEMPORAL (``TimeSeriesSplit``, 3 cortes).
4. Se añade una línea base ingenua ("el consumo es igual al de hace 24 h") para saber si
   el aprendizaje automático realmente aporta.
5. Métricas en el conjunto de prueba: MAE, RMSE, R² y MAPE (solo consumo real > 0.1 kWh).
6. El modelo seleccionado es el de menor RMSE promedio en validación cruzada temporal
   (``rmse_cv``), NO el que mejor salga en prueba, para no "espiar" el conjunto de prueba.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

OBJETIVO = "consumo_kwh"
NUMERICAS = ["hora", "dia_semana_num", "mes", "dia_anio", "es_fin_semana", "es_festivo", "hora_sin", "hora_cos",
             "mes_sin", "mes_cos", "temperatura_c", "humedad_pct", "temp_x_hora_pico", "superficie_m2",
             "num_ocupantes", "consumo_lag_1h", "consumo_lag_24h", "consumo_lag_168h", "consumo_media_movil_24h"]
CATEGORICAS = ["tipo_instalacion", "ciudad"]
CARACTERISTICAS = NUMERICAS + CATEGORICAS
N_ARBOLES_BUSQUEDA = 60      # árboles del RF durante la búsqueda (rápido)
N_ARBOLES_FINAL = 180        # árboles del RF final
MUESTRA_SVR = 4000
PALETA = dict(zip(config.TIPOS_INSTALACION, sns.color_palette("colorblind", 4)))


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------
def crear_preprocesador() -> ColumnTransformer:
    """Escalado de numéricas + OneHot de categóricas (se ajusta solo con entrenamiento)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])


def metricas(real, pred) -> dict:
    """MAE, RMSE, R² y MAPE (%) calculado solo donde el consumo real es > 0.1 kWh."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    m = real > 0.1
    return {"mae": float(mean_absolute_error(real, pred)),
            "rmse": float(np.sqrt(mean_squared_error(real, pred))),
            "r2": float(r2_score(real, pred)),
            "mape": float(np.mean(np.abs(real[m] - pred[m]) / real[m]) * 100)}


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_CONSUMO_PROC, encoding="utf-8")
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    # Orden global por tiempo: requisito de TimeSeriesSplit (los cortes respetan el tiempo)
    return df.sort_values(["fecha_hora", "id_instalacion"]).reset_index(drop=True)


def nombres_originales(pipe: Pipeline) -> list[str]:
    """Nombre de la variable original a la que pertenece cada columna transformada."""
    ct = pipe.named_steps["prep"]
    cat = ct.named_transformers_["cat"]
    nombres = list(NUMERICAS)
    for col, cats in zip(CATEGORICAS, cat.categories_):
        nombres += [col] * len(cats)
    return nombres


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------
def _guardar(nombre: str) -> None:
    plt.tight_layout()
    plt.savefig(config.DIR_FIGURAS / nombre, dpi=110)
    plt.close()


def figuras_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    # Perfil horario medio por tipo, laborable vs fin de semana
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, tipo in zip(ejes.ravel(), config.TIPOS_INSTALACION):
        d = df[df["tipo_instalacion"] == tipo]
        for etiqueta, sel, ls in (("lunes a viernes", d["es_fin_semana"] == 0, "-"), ("fin de semana", d["es_fin_semana"] == 1, "--")):
            perfil = d[sel].groupby("hora")[OBJETIVO].mean()
            ax.plot(perfil.index, perfil.values, ls, lw=2, color=PALETA[tipo], label=etiqueta)
        ax.set_title(tipo.capitalize())
        ax.set_ylabel("consumo medio (kWh/h)")
        ax.legend(fontsize=8)
    for ax in ejes[1]:
        ax.set_xlabel("hora del día")
    fig.suptitle("Perfil horario medio de consumo por tipo de instalación")
    _guardar("eda_consumo_hora_tipo.png")

    # Temperatura vs consumo relativo (se quita el efecto del perfil horario de cada instalación)
    rel = df[OBJETIVO] / df.groupby(["id_instalacion", "hora", "es_fin_semana"])[OBJETIVO].transform("mean")
    d = df.assign(rel=rel, bin_t=df["temperatura_c"].round())
    fig, ax = plt.subplots(figsize=(8, 5))
    for tipo in config.TIPOS_INSTALACION:
        g = d[d["tipo_instalacion"] == tipo].groupby("bin_t")["rel"].agg(["mean", "count"])
        g = g[g["count"] >= 30]
        ax.plot(g.index, g["mean"], marker="o", ms=4, lw=2, color=PALETA[tipo], label=tipo)
    ax.axvline(26, color="gray", ls=":", label="26 °C (inicia el aire acond.)")
    ax.set_xlabel("temperatura (°C, redondeada)")
    ax.set_ylabel("consumo / consumo típico de esa hora")
    ax.set_title("Efecto de la temperatura sobre el consumo")
    ax.legend()
    _guardar("eda_temp_vs_consumo.png")

    # Correlación de variables numéricas
    cols = [OBJETIVO, "temperatura_c", "humedad_pct", "hora", "dia_semana_num", "es_fin_semana", "es_festivo",
            "temp_x_hora_pico", "consumo_lag_1h", "consumo_lag_24h", "consumo_lag_168h", "consumo_media_movil_24h",
            "superficie_m2", "num_ocupantes"]
    fig, ax = plt.subplots(figsize=(10, 8))
    with warnings.catch_warnings():     # aviso interno de seaborn/matplotlib, sin efecto
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Matriz de correlación (Pearson)")
    _guardar("eda_correlacion.png")


def figuras_modelo(modelos: dict, pred: pd.DataFrame, importancias: dict) -> None:
    # Comparación de modelos
    tabla = pd.DataFrame(modelos).T
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, titulo) in zip(ejes, (("mae", "MAE (kWh) - menor es mejor"), ("rmse", "RMSE (kWh) - menor es mejor"),
                                       ("r2", "R² - mayor es mejor"))):
        orden = tabla[col].sort_values(ascending=(col != "r2"))
        ax.barh(orden.index, orden.values, color=sns.color_palette("colorblind", len(orden)))
        ax.invert_yaxis()
        ax.set_title(titulo)
        for i, v in enumerate(orden.values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.margins(x=0.2)
    _guardar("comparacion_modelos.png")

    # Real vs predicho
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for tipo in config.TIPOS_INSTALACION:
        d = pred[pred["tipo_instalacion"] == tipo]
        ax.scatter(d["real"].clip(lower=0.05), d["predicho"].clip(lower=0.05), s=6, alpha=0.4, color=PALETA[tipo], label=tipo)
    lim = [0.05, pred[["real", "predicho"]].to_numpy().max() * 1.2]
    ax.plot(lim, lim, "k--", lw=1, label="predicción perfecta")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("consumo real (kWh, escala log)"); ax.set_ylabel("consumo predicho (kWh, escala log)")
    ax.set_title("Real vs predicho (conjunto de prueba)")
    ax.legend(markerscale=3)
    _guardar("real_vs_predicho.png")

    # Importancia de variables
    imp = pd.Series(importancias).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(imp.index, imp.values, color=sns.color_palette("crest", len(imp)))
    ax.set_xlabel("importancia (Random Forest, suma de reducción de impureza)")
    ax.set_title("Importancia de variables")
    _guardar("importancia_variables.png")

    # Residuos
    rel = pred.loc[pred["real"] > 0.1, "error"] / pred.loc[pred["real"] > 0.1, "real"] * 100
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    ejes[0].hist(rel.clip(-60, 60), bins=60, color=sns.color_palette("colorblind")[0])
    ejes[0].axvline(0, color="k", lw=1)
    ejes[0].set_xlabel("error relativo (%) = (real - predicho) / real, recortado a ±60 %")
    ejes[0].set_ylabel("frecuencia"); ejes[0].set_title("Distribución del error relativo")
    for tipo in config.TIPOS_INSTALACION:
        d = pred[pred["tipo_instalacion"] == tipo]
        ejes[1].scatter(d["predicho"].clip(lower=0.05), d["error"], s=5, alpha=0.35, color=PALETA[tipo], label=tipo)
    ejes[1].axhline(0, color="k", lw=1); ejes[1].set_xscale("log")
    ejes[1].set_xlabel("consumo predicho (kWh, escala log)"); ejes[1].set_ylabel("residuo = real - predicho (kWh)")
    ejes[1].set_title("Residuos vs predicción"); ejes[1].legend(markerscale=3)
    _guardar("residuos.png")

    # Series de prueba (última semana) de tres instalaciones
    for id_inst in ("RES-001", "IND-001", "EDU-001"):
        d = pred[pred["id_instalacion"] == id_inst].sort_values("fecha_hora").tail(7 * 24)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(d["fecha_hora"], d["real"], color="k", lw=1.6, label="real")
        ax.plot(d["fecha_hora"], d["predicho"], color=sns.color_palette("colorblind")[1], lw=1.6, label="predicho")
        ax.set_ylabel("consumo (kWh/h)"); ax.set_title(f"{id_inst}: última semana del conjunto de prueba")
        ax.legend(); fig.autofmt_xdate()
        _guardar(f"serie_prueba_{id_inst}.png")


# --------------------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------------------
def main() -> dict:
    t_total = time.time()
    df = cargar_datos()
    figuras_eda(df)
    ent, pru = df[df["conjunto"] == "entrenamiento"], df[df["conjunto"] == "prueba"]
    X_ent, y_ent, X_pru, y_pru = ent[CARACTERISTICAS], ent[OBJETIVO], pru[CARACTERISTICAS], pru[OBJETIVO]
    cv = TimeSeriesSplit(n_splits=3)
    sem = config.SEMILLA

    def pipe(modelo):
        return Pipeline([("prep", crear_preprocesador()), ("modelo", modelo)])

    # (nombre, pipeline, rejilla de hiperparámetros o None)
    candidatos = [
        ("Regresión Lineal", pipe(LinearRegression()), None),
        ("Árbol de Decisión", pipe(DecisionTreeRegressor(random_state=sem)),
         {"modelo__max_depth": [6, 10, 16], "modelo__min_samples_leaf": [5, 20]}),
        ("KNN", pipe(KNeighborsRegressor()),
         {"modelo__n_neighbors": [5, 10, 20], "modelo__weights": ["distance"]}),
        ("Random Forest", pipe(RandomForestRegressor(n_estimators=N_ARBOLES_BUSQUEDA, n_jobs=-1, random_state=sem)),
         {"modelo__max_depth": [14, None], "modelo__min_samples_leaf": [3, 8], "modelo__max_features": [0.5, 1.0]}),
        ("SVR (muestra)", pipe(TransformedTargetRegressor(SVR(C=10, epsilon=0.02), transformer=StandardScaler())), None),
    ]

    log = ["EVIDENCIA - ENTRENAMIENTO DE MODELOS DE CONSUMO (SIAPE)",
           f"Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M:%S}  |  Semilla: {sem}",
           f"Entrenamiento: {len(ent)} filas ({ent['fecha_hora'].min()} a {ent['fecha_hora'].max()})",
           f"Prueba: {len(pru)} filas ({pru['fecha_hora'].min()} a {pru['fecha_hora'].max()})",
           f"Características ({len(CARACTERISTICAS)}): {', '.join(CARACTERISTICAS)}",
           "Validación cruzada: TimeSeriesSplit(3 cortes, ventana creciente); métrica rmse_cv = RMSE promedio",
           ""]
    resultados, hiper, finales, tiempos_busqueda = {}, {}, {}, {}
    muestra_svr = ent.iloc[np.linspace(0, len(ent) - 1, MUESTRA_SVR).astype(int)]   # muestra sistemática en el tiempo

    for nombre, modelo, rejilla in candidatos:
        t0 = time.time()
        X_aj, y_aj = (muestra_svr[CARACTERISTICAS], muestra_svr[OBJETIVO]) if "SVR" in nombre else (X_ent, y_ent)
        if rejilla:
            gs = GridSearchCV(modelo, rejilla, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
            gs.fit(X_aj, y_aj)
            rmse_cv = -gs.best_score_
            params = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}
            modelo = gs.best_estimator_.set_params(
                **({"modelo__n_estimators": N_ARBOLES_FINAL} if nombre == "Random Forest" else {}))
        else:
            rmse_cv = float(-cross_val_score(modelo, X_aj, y_aj, cv=cv, scoring="neg_root_mean_squared_error").mean())
            params = {}
        tiempos_busqueda[nombre] = time.time() - t0
        t1 = time.time()
        modelo.fit(X_aj, y_aj)                                   # ajuste final con todo el entrenamiento
        t_fit = time.time() - t1
        pred = modelo.predict(X_pru)
        m = metricas(y_pru, pred)
        m.update({"rmse_cv": float(rmse_cv), "tiempo_s": round(t_fit, 2), "tiempo_busqueda_s": round(tiempos_busqueda[nombre], 2)})
        resultados[nombre], hiper[nombre], finales[nombre] = m, params, (modelo, pred)
        print(f"  {nombre:<20} RMSE={m['rmse']:.3f} R2={m['r2']:.4f} rmse_cv={rmse_cv:.3f} ({tiempos_busqueda[nombre]:.0f}s)")

    # Línea base ingenua: el consumo de hoy a esta hora = el de hace 24 h
    base = metricas(y_pru, pru["consumo_lag_24h"])
    base.update({"rmse_cv": None, "tiempo_s": 0.0, "tiempo_busqueda_s": 0.0})
    resultados["Línea base (hace 24 h)"] = base

    # Selección por validación cruzada temporal
    candidatos_validos = {k: v for k, v in resultados.items() if v["rmse_cv"] is not None}
    seleccionado = min(candidatos_validos, key=lambda k: candidatos_validos[k]["rmse_cv"])
    modelo_sel, pred_sel = finales[seleccionado]
    mejor_prueba = min(candidatos_validos, key=lambda k: candidatos_validos[k]["rmse"])

    # Predicciones de prueba
    pred_df = pd.DataFrame({
        "id_instalacion": pru["id_instalacion"].values,
        "fecha_hora": pru["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "tipo_instalacion": pru["tipo_instalacion"].values,
        "real": y_pru.values, "predicho": np.round(np.clip(pred_sel, 0, None), 3)})
    pred_df["error"] = np.round(pred_df["real"] - pred_df["predicho"], 3)   # residuo = real - predicho
    pred_df.to_csv(config.DIR_OUTPUTS / "predicciones_test.csv", index=False, encoding="utf-8")
    pred_fig = pred_df.assign(fecha_hora=pd.to_datetime(pred_df["fecha_hora"]))

    # Métricas por tipo de instalación (modelo seleccionado)
    por_tipo = {}
    for tipo in config.TIPOS_INSTALACION:
        d = pred_df[pred_df["tipo_instalacion"] == tipo]
        mt = metricas(d["real"], d["predicho"])
        por_tipo[tipo] = {"mae": mt["mae"], "r2": mt["r2"], "rmse": mt["rmse"], "mape": mt["mape"]}

    # Importancias (solo RF las tiene) agregadas a las variables originales
    rf_pipe = finales["Random Forest"][0]
    imp_cruda = rf_pipe.named_steps["modelo"].feature_importances_
    imp = pd.Series(imp_cruda).groupby(nombres_originales(rf_pipe)).sum().sort_values(ascending=False)
    importancias = {k: float(v) for k, v in imp.items()}

    # Intervalo con percentiles de los árboles (5-95): cobertura empírica en prueba
    cobertura = None
    if seleccionado == "Random Forest":
        Xt = modelo_sel.named_steps["prep"].transform(X_pru)
        por_arbol = np.array([t.predict(Xt) for t in modelo_sel.named_steps["modelo"].estimators_])
        lo, hi = np.percentile(por_arbol, 5, axis=0), np.percentile(por_arbol, 95, axis=0)
        cobertura = float(np.mean((y_pru.values >= lo) & (y_pru.values <= hi)))

    hiper_sel = hiper[seleccionado]
    if seleccionado == "Random Forest":
        hiper_sel = {**hiper_sel, "n_estimators": N_ARBOLES_FINAL}
    salida = {
        "seleccionado": seleccionado,
        "tipo_aprendizaje": "supervisado - regresión",
        "objetivo": OBJETIVO,
        "caracteristicas": CARACTERISTICAS,
        "hiperparametros": hiper_sel,
        "modelos": resultados,
        "metricas_por_tipo": por_tipo,
        "importancias": importancias,
        "n_entrenamiento": int(len(ent)),
        "n_prueba": int(len(pru)),
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hiperparametros_por_modelo": hiper,
        "criterio_seleccion": "menor rmse_cv (validación cruzada temporal, TimeSeriesSplit 3)",
        "cobertura_intervalo_90": cobertura,
    }
    joblib.dump(modelo_sel, config.MODELO_CONSUMO, compress=3)
    config.METRICAS_ML.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    figuras_modelo({k: v for k, v in resultados.items()}, pred_fig, importancias)

    # Evidencia
    log.append(f"{'Modelo':<24}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'MAPE %':>9}{'rmse_cv':>10}{'fit s':>8}{'búsq. s':>9}")
    for k, v in resultados.items():
        cvs = f"{v['rmse_cv']:.3f}" if v["rmse_cv"] is not None else "-"
        log.append(f"{k:<24}{v['mae']:>9.3f}{v['rmse']:>9.3f}{v['r2']:>9.4f}{v['mape']:>9.2f}{cvs:>10}{v['tiempo_s']:>8.1f}{v['tiempo_busqueda_s']:>9.1f}")
    log += ["", "Hiperparámetros elegidos por búsqueda (GridSearchCV + TimeSeriesSplit):"]
    log += [f"  {k}: {v if v else '(sin búsqueda; valores por defecto)'}" for k, v in hiper.items()]
    log += [f"  Random Forest final: n_estimators={N_ARBOLES_FINAL} (búsqueda con {N_ARBOLES_BUSQUEDA})",
            f"  SVR se entrena con una muestra sistemática de {MUESTRA_SVR} filas; los demás con las {len(ent)}.",
            "", f"MODELO SELECCIONADO: {seleccionado} (menor rmse_cv).",
            f"Mejor RMSE en prueba entre los modelos de ML: {mejor_prueba}."]
    log += ["", f"Métricas por tipo de instalación ({seleccionado}):"]
    log += [f"  {t:<12} MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}  R2={v['r2']:.4f}  MAPE={v['mape']:.2f} %" for t, v in por_tipo.items()]
    log += ["", "Importancia de variables (Random Forest, agregada por variable original):"]
    log += [f"  {k:<26}{v:.4f}" for k, v in importancias.items()]
    if cobertura is not None:
        log += ["", f"Intervalo de percentiles 5-95 de los árboles: cobertura empírica en prueba = {cobertura:.1%} (nominal 90 %)."]
    log += ["", "Anti-fuga: lags calculados solo con valores pasados (preparar_datos.py); división temporal;",
            "escalado y OneHot dentro del Pipeline; selección del modelo con validación cruzada temporal.",
            f"Modelo guardado en: {config.MODELO_CONSUMO}", f"Métricas en: {config.METRICAS_ML}",
            f"Predicciones de prueba: {config.DIR_OUTPUTS / 'predicciones_test.csv'}",
            f"Tiempo total de ejecución: {time.time() - t_total:.0f} s"]
    texto = "\n".join(log)
    print(texto)
    (config.DIR_EVIDENCIAS / "entrenamiento_modelo.txt").write_text(texto + "\n", encoding="utf-8")
    return salida


if __name__ == "__main__":
    main()
