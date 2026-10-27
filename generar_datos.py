"""Genera un conjunto de datos SINTÉTICO de participación electoral ciudadana.

El proceso generador tiene dos etapas:
  1. Cada persona pertenece a un arquetipo latente (p. ej. "jóvenes universitarios",
     "comunidades rurales alejadas") que define la distribución de sus atributos.
  2. La decisión de votar en cada elección sale de un modelo logístico sobre esos
     atributos (edad, escolaridad, interés político, distancia a la casilla, ...) más un
     efecto individual persistente (el "hábito de votar").

Los arquetipos NO se guardan en el archivo principal: el análisis tiene que
redescubrirlos solo a partir del comportamiento y las características observables.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2024
N = 30_000
ANIO_REF = 2024
# Participación objetivo entre quienes eran elegibles (>= 18 años) en cada elección.
ELECCIONES = {2018: 0.63, 2021: 0.52, 2024: 0.61}

REGIONES = ["Norte", "Bajío", "Centro", "Golfo", "Sur"]
EFECTO_REGION = {"Norte": -0.10, "Bajío": 0.05, "Centro": 0.20, "Golfo": -0.15, "Sur": 0.0}
ZONAS = ["urbana", "semiurbana", "rural"]
REGION_POR_ZONA = {
    "urbana": [0.20, 0.20, 0.35, 0.12, 0.13],
    "semiurbana": [0.18, 0.20, 0.25, 0.15, 0.22],
    "rural": [0.10, 0.15, 0.15, 0.20, 0.40],
}

# edad=(min, max) · esc=(media, sd, min, max) · ingreso=(mediana USD/mes, sigma log)
# zona=(p urbana, p semiurbana, p rural) · interes/conf=(media, sd) en escala 1-5
# dist=(mediana km, sigma log) · redes=(media, sd) horas/día · contacto, org = prob.
ARQUETIPOS = [
    dict(nombre="jovenes_universitarios", peso=0.15, edad=(18, 29), esc=(15.0, 1.5, 12, 20),
         ingreso=(650, 0.45), zona=(0.85, 0.12, 0.03), interes=(3.3, 0.8), conf=(2.5, 0.7),
         dist=(0.8, 0.5), redes=(6.5, 1.2), contacto=0.20, org=0.20,
         ocup={"Estudiante": 0.60, "Empleo formal": 0.28, "Trabajo independiente o informal": 0.12}),
    dict(nombre="jovenes_vulnerables", peso=0.17, edad=(18, 29), esc=(7.5, 1.8, 3, 10),
         ingreso=(180, 0.45), zona=(0.25, 0.30, 0.45), interes=(1.8, 0.7), conf=(2.2, 0.8),
         dist=(3.5, 0.7), redes=(3.0, 1.3), contacto=0.08, org=0.03,
         ocup={"Estudiante": 0.15, "Trabajo independiente o informal": 0.50,
               "Hogar y cuidados": 0.15, "Campo y agricultura": 0.20}),
    dict(nombre="adultos_sin_tiempo", peso=0.18, edad=(30, 55), esc=(10.5, 1.8, 7, 14),
         ingreso=(350, 0.40), zona=(0.75, 0.20, 0.05), interes=(2.5, 0.8), conf=(1.9, 0.7),
         dist=(2.0, 0.6), redes=(2.8, 1.0), contacto=0.10, org=0.04,
         ocup={"Empleo formal": 0.50, "Trabajo independiente o informal": 0.35,
               "Hogar y cuidados": 0.15}),
    dict(nombre="mayores_comprometidos", peso=0.12, edad=(62, 85), esc=(6.0, 2.5, 0, 12),
         ingreso=(250, 0.45), zona=(0.50, 0.25, 0.25), interes=(4.2, 0.6), conf=(4.0, 0.7),
         dist=(2.0, 0.7), redes=(0.5, 0.5), contacto=0.60, org=0.20,
         ocup={"Jubilación o pensión": 0.70, "Hogar y cuidados": 0.20, "Campo y agricultura": 0.10}),
    dict(nombre="rurales_alejados", peso=0.12, edad=(30, 70), esc=(4.5, 2.0, 0, 9),
         ingreso=(150, 0.40), zona=(0.0, 0.10, 0.90), interes=(2.9, 0.8), conf=(3.0, 0.8),
         dist=(12.0, 0.4), redes=(0.5, 0.5), contacto=0.08, org=0.08,
         ocup={"Campo y agricultura": 0.65, "Hogar y cuidados": 0.20,
               "Trabajo independiente o informal": 0.15}),
    dict(nombre="profesionistas_desencantados", peso=0.07, edad=(30, 60), esc=(17.5, 1.2, 15, 20),
         ingreso=(2000, 0.35), zona=(0.92, 0.07, 0.01), interes=(3.6, 0.7), conf=(1.5, 0.5),
         dist=(1.2, 0.5), redes=(3.8, 1.0), contacto=0.12, org=0.12,
         ocup={"Profesionista": 0.75, "Empleo formal": 0.15, "Trabajo independiente o informal": 0.10}),
    dict(nombre="ciudadania_organizada", peso=0.06, edad=(28, 65), esc=(14.0, 2.2, 9, 20),
         ingreso=(650, 0.45), zona=(0.65, 0.20, 0.15), interes=(4.7, 0.4), conf=(3.2, 0.8),
         dist=(1.8, 0.7), redes=(3.3, 1.2), contacto=0.65, org=0.95,
         ocup={"Profesionista": 0.40, "Empleo formal": 0.35, "Trabajo independiente o informal": 0.25}),
    dict(nombre="mayores_rurales_aislados", peso=0.07, edad=(60, 85), esc=(2.5, 1.8, 0, 7),
         ingreso=(130, 0.40), zona=(0.0, 0.10, 0.90), interes=(3.0, 0.8), conf=(3.4, 0.8),
         dist=(11.0, 0.4), redes=(0.2, 0.3), contacto=0.12, org=0.05,
         ocup={"Campo y agricultura": 0.50, "Jubilación o pensión": 0.25, "Hogar y cuidados": 0.25}),
    dict(nombre="jovenes_activistas_digitales", peso=0.06, edad=(19, 32), esc=(15.5, 1.8, 11, 20),
         ingreso=(500, 0.50), zona=(0.90, 0.08, 0.02), interes=(4.5, 0.5), conf=(1.5, 0.6),
         dist=(1.0, 0.5), redes=(8.0, 1.3), contacto=0.30, org=0.60,
         ocup={"Estudiante": 0.40, "Empleo formal": 0.30, "Trabajo independiente o informal": 0.20,
               "Profesionista": 0.10}),
]


def _muestrear_arquetipo(rng, a, m):
    esc_mu, esc_sd, esc_lo, esc_hi = a["esc"]
    return pd.DataFrame({
        "edad": rng.integers(a["edad"][0], a["edad"][1] + 1, m),
        "escolaridad_anios": np.rint(np.clip(rng.normal(esc_mu, esc_sd, m), esc_lo, esc_hi)).astype(int),
        "ingreso_mensual_usd": np.clip(rng.lognormal(np.log(a["ingreso"][0]), a["ingreso"][1], m), 60, 8000),
        "zona": rng.choice(ZONAS, m, p=a["zona"]),
        "interes_politico": np.rint(np.clip(rng.normal(*a["interes"], m), 1, 5)).astype(int),
        "confianza_instituciones": np.rint(np.clip(rng.normal(*a["conf"], m), 1, 5)).astype(int),
        "distancia_casilla_km": np.round(np.clip(rng.lognormal(np.log(a["dist"][0]), a["dist"][1], m), 0.1, 40), 1),
        "horas_redes_dia": np.round(np.clip(rng.normal(*a["redes"], m), 0, 12), 1),
        "contacto_campana": (rng.random(m) < a["contacto"]).astype(int),
        "participa_org_civil": (rng.random(m) < a["org"]).astype(int),
        "ocupacion": rng.choice(list(a["ocup"]), m, p=list(a["ocup"].values())),
    })


def _logit_base(df, edad, efecto_individual):
    """Log-odds de votar sin el desplazamiento propio de cada elección."""
    a = edad - 45
    z = 0.04 * a - 0.0006 * a ** 2
    z = z + 0.07 * (df["escolaridad_anios"] - 10)
    z = z + 0.35 * (np.log(df["ingreso_mensual_usd"]) - np.log(400))
    z = z + 0.55 * (df["interes_politico"] - 3) + 0.35 * (df["confianza_instituciones"] - 3)
    z = z - 0.28 * np.log1p(df["distancia_casilla_km"])
    z = z + 0.50 * df["contacto_campana"] + 0.45 * df["participa_org_civil"]
    z = z + 0.05 * (df["horas_redes_dia"] - 2)
    z = z + 0.10 * (df["sexo"] == "Mujer")
    z = z + df["region"].map(EFECTO_REGION)
    return (z + efecto_individual).to_numpy(dtype=float)


def _ajustar_desplazamiento(base, objetivo):
    """Busca el desplazamiento b tal que la media de sigmoide(base + b) sea `objetivo`."""
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(base + mid)))).mean() < objetivo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generar(n=N, seed=SEED):
    """Devuelve (datos, arquetipo_verdadero). `arquetipo_verdadero` es solo para validar."""
    rng = np.random.default_rng(seed)
    pesos = np.array([a["peso"] for a in ARQUETIPOS])
    assert abs(pesos.sum() - 1) < 1e-9
    tamanos = rng.multinomial(n, pesos)

    partes, etiquetas = [], []
    for a, m in zip(ARQUETIPOS, tamanos):
        partes.append(_muestrear_arquetipo(rng, a, m))
        etiquetas += [a["nombre"]] * int(m)
    df = pd.concat(partes, ignore_index=True)
    arq = pd.Series(etiquetas, name="arquetipo")

    orden = rng.permutation(n)
    df = df.iloc[orden].reset_index(drop=True)
    arq = arq.iloc[orden].reset_index(drop=True)

    df["sexo"] = rng.choice(["Mujer", "Hombre"], n, p=[0.52, 0.48])
    df["region"] = ""
    for zona, probs in REGION_POR_ZONA.items():
        idx = df.index[df["zona"] == zona]
        df.loc[idx, "region"] = rng.choice(REGIONES, len(idx), p=probs)

    u = pd.Series(rng.normal(0, 0.7, n))  # hábito individual persistente
    for anio, objetivo in ELECCIONES.items():
        edad_e = df["edad"] - (ANIO_REF - anio)
        elegible = (edad_e >= 18).to_numpy()
        base = _logit_base(df, edad_e, u)
        b = _ajustar_desplazamiento(base[elegible], objetivo)
        p = 1 / (1 + np.exp(-(base + b)))
        voto = rng.random(n) < p
        serie = pd.Series(pd.array(voto.astype(int), dtype="Int8"))
        df[f"voto_{anio}"] = serie.mask(~elegible)

    # Datos faltantes (como en una encuesta real)
    df.loc[rng.random(n) < 0.03, "ingreso_mensual_usd"] = np.nan
    conf = df["confianza_instituciones"].astype("Int8")
    df["confianza_instituciones"] = conf.mask(pd.Series(rng.random(n) < 0.02))
    df["ingreso_mensual_usd"] = df["ingreso_mensual_usd"].round(0)

    df.insert(0, "id", [f"C{i:06d}" for i in range(1, n + 1)])
    columnas = ["id", "edad", "sexo", "region", "zona", "escolaridad_anios", "ingreso_mensual_usd",
                "ocupacion", "interes_politico", "confianza_instituciones", "distancia_casilla_km",
                "horas_redes_dia", "contacto_campana", "participa_org_civil",
                "voto_2018", "voto_2021", "voto_2024"]
    return df[columnas], arq


def guardar(carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    df, arq = generar()
    df.to_csv(carpeta / "ciudadanos.csv", index=False, encoding="utf-8")
    arq.to_frame().assign(id=df["id"]).to_csv(carpeta / "_verdad_sintetica.csv", index=False)
    return df, arq


if __name__ == "__main__":
    datos, _ = guardar(Path(__file__).parent / "data")
    print(f"{len(datos):,} ciudadanos generados. Participación 2024: {datos['voto_2024'].mean():.1%}")
