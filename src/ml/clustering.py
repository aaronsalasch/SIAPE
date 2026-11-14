"""Aprendizaje NO SUPERVISADO: agrupación de perfiles diarios de consumo con K-Means.

Ejecución:  python -m src.ml.clustering

Idea: cada instalación-día se describe con un vector de 24 valores (el consumo de cada
hora). Para comparar la FORMA de la curva y no el tamaño de la instalación, cada vector se
divide entre su propio promedio (perfil normalizado: media = 1). K-Means agrupa los días
con forma parecida (p. ej. "pico vespertino residencial", "meseta industrial",
"día cerrado").

Elección de k (2..8) con dos criterios:
  * Codo: inercia (suma de distancias al cuadrado al centroide) y el punto de máxima
    curvatura (distancia a la recta que une los extremos de la curva).
  * Silhouette: qué tan separados están los grupos (entre -1 y 1; más alto es mejor).
Regla: entre k >= 4 (hay 4 tipos de instalación con formas distintas; con k menor se
mezclarían tipos) se toma el máximo silhouette y, si un k mayor queda a menos de
``TOLERANCIA_SIL`` (0.01) de ese máximo, se prefiere el mayor (separa grupos con sentido
operativo, como los días cerrados de las escuelas). También se reporta el codo.

Salidas: ``config.MODELO_KMEANS``, ``models/clusters_resumen.json``, figuras
``kmeans_codo_silhouette.png`` y ``kmeans_centroides.png`` y la evidencia ``clustering.txt``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

K_MIN, K_MAX = 2, 8
K_MIN_ELEGIBLE = 4
TOLERANCIA_SIL = 0.01
RUTA_RESUMEN = config.DIR_MODELOS / "clusters_resumen.json"


def construir_perfiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (perfiles normalizados, perfiles en kWh). Índice = (id_instalacion, fecha)."""
    df = pd.read_csv(config.CSV_CONSUMO_PROC, usecols=["id_instalacion", "tipo_instalacion", "fecha_hora", "hora", "consumo_kwh"])
    df["fecha"] = df["fecha_hora"].str[:10]
    crudo = df.pivot_table(index=["id_instalacion", "fecha"], columns="hora", values="consumo_kwh", aggfunc="mean")
    crudo = crudo.dropna()                                  # solo días con las 24 horas completas
    crudo.columns = [f"h{h:02d}" for h in crudo.columns]
    normalizado = crudo.div(crudo.mean(axis=1), axis=0)     # media del día = 1
    return normalizado, crudo


def _codo(ks, inercias) -> int:
    """k del 'codo': el punto más alejado de la recta que une el primero y el último."""
    x, y = np.array(ks, float), np.array(inercias, float)
    x, y = (x - x[0]) / (x[-1] - x[0]), (y - y[-1]) / (y[0] - y[-1])
    return int(ks[int(np.argmax(np.abs(x + y - 1) / np.sqrt(2)))])


def main() -> dict:
    sns.set_theme(style="whitegrid", context="notebook")
    perfiles, crudo = construir_perfiles()
    X = perfiles.to_numpy()
    tipo_de = {i["id"]: i["tipo"] for i in config.INSTALACIONES}

    ks = list(range(K_MIN, K_MAX + 1))
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEMILLA).fit(X)
        inercias.append(float(km.inertia_))
        siluetas.append(float(silhouette_score(X, km.labels_)))
    k_codo = _codo(ks, inercias)
    elegibles = [(s, k) for k, s in zip(ks, siluetas) if k >= K_MIN_ELEGIBLE]
    mejor_sil = max(elegibles)[0]
    k_sel = max(k for s, k in elegibles if s >= mejor_sil - TOLERANCIA_SIL)

    km = KMeans(n_clusters=k_sel, n_init=10, random_state=config.SEMILLA).fit(X)
    etiquetas = km.labels_
    dist = np.linalg.norm(X - km.cluster_centers_[etiquetas], axis=1)
    sil = float(silhouette_score(X, etiquetas))

    perfiles_out = perfiles.copy()
    perfiles_out["cluster"], perfiles_out["distancia"] = etiquetas, dist
    perfiles_out = perfiles_out.reset_index()
    perfiles_out["tipo"] = perfiles_out["id_instalacion"].map(tipo_de)

    # Descripción automática de cada cluster
    horas = np.arange(24)
    descripciones, distribucion = {}, {}
    for c in range(k_sel):
        cen = km.cluster_centers_[c]
        sub = perfiles_out[perfiles_out["cluster"] == c]
        tipos = sub["tipo"].value_counts()
        distribucion[str(c)] = {t: int(tipos.get(t, 0)) for t in config.TIPOS_INSTALACION}
        pico = int(horas[np.argmax(cen)])
        variab = float(cen.std())
        forma = ("curva casi plana (sin pico marcado: operación continua o cerrado)" if variab < 0.15 else
                 "consumo muy bajo casi todo el día (cerrado/ausencia)" if cen.max() < 1.6 and cen.min() < 0.3 else
                 "con pico marcado")
        descripciones[str(c)] = (f"{len(sub)} días; tipo dominante: {tipos.index[0]} ({100 * tipos.iloc[0] / len(sub):.0f} %); "
                                 f"hora pico: {pico}:00; {forma}")

    atip = perfiles_out.sort_values("distancia", ascending=False).head(20)
    dias_atipicos = [{"id_instalacion": r.id_instalacion, "tipo": r.tipo, "fecha": r.fecha, "cluster": int(r.cluster),
                      "distancia": round(float(r.distancia), 4),
                      "consumo_medio_kwh": round(float(crudo.loc[(r.id_instalacion, r.fecha)].mean()), 3)}
                     for r in atip.itertuples()]

    resumen = {
        "k": int(k_sel), "silhouette": sil,
        "k_codo": k_codo, "regla_eleccion": f"k >= {K_MIN_ELEGIBLE}; mayor k con silhouette a menos de {TOLERANCIA_SIL} del máximo",
        "inercias": {str(k): v for k, v in zip(ks, inercias)},
        "silhouettes": {str(k): v for k, v in zip(ks, siluetas)},
        "centroides": {str(c): [round(float(v), 4) for v in km.cluster_centers_[c]] for c in range(k_sel)},
        "descripciones": descripciones,
        "tamanos": {str(c): int((etiquetas == c).sum()) for c in range(k_sel)},
        "distribucion_por_tipo": distribucion,
        "dias_atipicos": dias_atipicos,
        "n_perfiles": int(len(X)),
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(km, config.MODELO_KMEANS, compress=3)
    RUTA_RESUMEN.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Figuras
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    azul, naranja = sns.color_palette("colorblind")[0], sns.color_palette("colorblind")[1]
    ejes[0].plot(ks, inercias, "o-", color=azul, lw=2)
    ejes[0].axvline(k_codo, color="gray", ls=":", label=f"codo (k={k_codo})")
    ejes[0].set_xlabel("k"); ejes[0].set_ylabel("inercia"); ejes[0].set_title("Método del codo"); ejes[0].legend()
    ejes[1].plot(ks, siluetas, "o-", color=naranja, lw=2)
    ejes[1].axvline(k_sel, color="gray", ls=":", label=f"elegido (k={k_sel})")
    ejes[1].set_xlabel("k"); ejes[1].set_ylabel("silhouette"); ejes[1].set_title("Coeficiente de silhouette"); ejes[1].legend()
    plt.tight_layout(); plt.savefig(config.DIR_FIGURAS / "kmeans_codo_silhouette.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c, color in zip(range(k_sel), sns.color_palette("colorblind", k_sel)):
        tipos = distribucion[str(c)]
        dom = max(tipos, key=tipos.get)
        ax.plot(horas, km.cluster_centers_[c], lw=2.2, color=color,
                label=f"cluster {c} (n={resumen['tamanos'][str(c)]}, dom.: {dom})")
    ax.set_xlabel("hora del día"); ax.set_ylabel("consumo / promedio del día")
    ax.set_title(f"Centroides de K-Means (k={k_sel}): perfiles diarios típicos")
    ax.set_xticks(range(0, 24, 2)); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(config.DIR_FIGURAS / "kmeans_centroides.png", dpi=110); plt.close()

    # ---- Evidencia
    log = ["EVIDENCIA - CLUSTERING DE PERFILES DIARIOS (K-MEANS, SIAPE)",
           f"Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M:%S}  |  Semilla: {config.SEMILLA}",
           f"Perfiles instalación-día completos (24 h): {len(X)}  |  Vector: 24 horas / promedio del día", "",
           f"{'k':>3}{'inercia':>12}{'silhouette':>12}"]
    log += [f"{k:>3}{i:>12.2f}{s:>12.4f}" for k, i, s in zip(ks, inercias, siluetas)]
    log += ["", f"Codo (máxima curvatura): k={k_codo}",
            f"Regla: k >= {K_MIN_ELEGIBLE}, mayor k con silhouette a menos de {TOLERANCIA_SIL} del máximo  ->  k elegido = {k_sel}, silhouette = {sil:.4f}", "",
            "Cluster: tamaño, distribución por tipo y descripción"]
    for c in range(k_sel):
        log.append(f"  Cluster {c}: n={resumen['tamanos'][str(c)]}  tipos={distribucion[str(c)]}")
        log.append(f"      {descripciones[str(c)]}")
    log += ["", "Top 20 días más atípicos (más lejanos a su centroide):",
            f"{'instalación':<12}{'fecha':<12}{'cluster':>8}{'distancia':>11}{'kWh medio':>11}"]
    log += [f"{d['id_instalacion']:<12}{d['fecha']:<12}{d['cluster']:>8}{d['distancia']:>11.3f}{d['consumo_medio_kwh']:>11.2f}"
            for d in dias_atipicos]
    log += ["", f"Modelo: {config.MODELO_KMEANS}", f"Resumen: {RUTA_RESUMEN}"]
    texto = "\n".join(log)
    print(texto)
    (config.DIR_EVIDENCIAS / "clustering.txt").write_text(texto + "\n", encoding="utf-8")
    return resumen


if __name__ == "__main__":
    main()
