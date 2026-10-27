"""Verificaciones automáticas: integridad de datos y paridad Python <-> JavaScript del simulador."""
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import analisis as A

RAIZ = Path(__file__).parent
fallos = []


def comprobar(cond, msg):
    print(("OK   " if cond else "FALLA"), msg)
    if not cond:
        fallos.append(msg)


def main():
    df, _ = A.cargar(RAIZ / "data" / "ciudadanos.csv")
    out = pd.read_csv(RAIZ / "salida" / "ciudadanos_con_perfil.csv")
    html_txt = (RAIZ / "salida" / "reporte.html").read_text(encoding="utf-8")
    D = json.loads(re.search(r"const DATOS = (.*?);</script>", html_txt, re.S).group(1).replace("<\\/", "</"))

    # --- Integridad de datos
    comprobar(df["id"].is_unique and len(df) == len(out), "ids únicos y mismo tamaño en las salidas")
    comprobar((df["id"].to_numpy() == out["id"].to_numpy()).all(), "las salidas conservan el orden de ids")
    comprobar(df["voto_2024"].notna().all(), "todos eran elegibles en 2024")
    elegible_2018 = (df["edad"] - 6) >= 18
    comprobar(bool((df["voto_2018"].notna() == elegible_2018).all()), "elegibilidad 2018 coincide con edad >= 18")
    elegible_2021 = (df["edad"] - 3) >= 18
    comprobar(bool((df["voto_2021"].notna() == elegible_2021).all()), "elegibilidad 2021 coincide con edad >= 18")
    comprobar(bool(df.drop(columns=["voto_2018", "voto_2021"]).notna().all().all()), "sin faltantes tras la imputación")
    comprobar(sorted(out["perfil"].unique()) == list(range(1, D["perfiles"].__len__() + 1)), "perfiles numerados 1..k")
    comprobar(abs(sum(p["share"] for p in D["perfiles"]) - 1) < 1e-9, "las proporciones de perfil suman 1")
    comprobar(bool(out["prob_participacion"].between(0, 1).all()), "probabilidades en [0, 1]")
    por_perfil = out.groupby("perfil")["voto_2024"].mean().to_numpy()
    comprobar(bool(np.all(np.diff(por_perfil) <= 1e-12)), "perfiles ordenados de mayor a menor participación 2024")
    nombres = [p["nombre"] for p in D["perfiles"]]
    comprobar(len(set(nombres)) == len(nombres), "nombres de perfil únicos")

    # --- Paridad JS <-> Python
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), 400, replace=False)
    muestra = df.iloc[idx]
    cols = ["edad", "sexo", "region", "zona", "escolaridad_anios", "ingreso_mensual_usd", "interes_politico",
            "confianza_instituciones", "distancia_casilla_km", "horas_redes_dia", "contacto_campana",
            "participa_org_civil", "tasa_participacion"]
    regs = json.loads(muestra[cols].to_json(orient="records"))
    js = (RAIZ / "static" / "app.js").read_text(encoding="utf-8")
    harness = (
        "const A=require(%s);const D=%s;const regs=%s;"
        "const res=regs.map(r=>({p:A.probabilidad(D.modelo,r),perfil:A.perfilMasCercano(D.cluster,r),"
        "pal:Object.fromEntries(D.palancas.map(l=>[l.id,A.probabilidad(D.modelo,A.PALANCAS[l.id](r))]))}));"
        "console.log(JSON.stringify(res));"
    )
    with tempfile.TemporaryDirectory() as tmp:
        ruta_app = Path(tmp) / "app.js"
        ruta_app.write_text(js, encoding="utf-8")
        ruta_h = Path(tmp) / "h.js"
        ruta_h.write_text(harness % (json.dumps(str(ruta_app)), json.dumps(D), json.dumps(regs)), encoding="utf-8")
        res = json.loads(subprocess.run(["node", str(ruta_h)], capture_output=True, text=True, check=True).stdout)

    p_js = np.array([r["p"] for r in res])
    comprobar(float(np.abs(p_js - out["prob_participacion"].to_numpy()[idx]).max()) < 1e-9,
              "JS reproduce la probabilidad de Python (400 registros)")
    perfil_js = np.array([r["perfil"] for r in res]) + 1
    comprobar(bool((perfil_js == out["perfil"].to_numpy()[idx]).all()), "JS asigna el mismo perfil que K-Means")
    for lv in A.LEVERS:
        esperado = A.predecir(D["modelo"], lv["f"](muestra))
        obtenido = np.array([r["pal"][lv["id"]] for r in res])
        comprobar(float(np.abs(esperado - obtenido).max()) < 1e-9, f"JS reproduce la palanca '{lv['id']}'")

    print("\n" + ("Todo correcto." if not fallos else f"{len(fallos)} verificación(es) fallaron."))
    raise SystemExit(1 if fallos else 0)


if __name__ == "__main__":
    main()
