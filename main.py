"""Ejecuta todo el flujo: datos sintéticos -> análisis -> tablero HTML."""
from pathlib import Path

import pandas as pd

import analisis
import generar_datos
import reporte

RAIZ = Path(__file__).parent


def main():
    print("1/3 Generando datos sintéticos...")
    generar_datos.guardar(RAIZ / "data")

    print("2/3 Analizando (perfiles, modelo, escenarios)...")
    R = analisis.ejecutar(RAIZ / "data" / "ciudadanos.csv", RAIZ / "data" / "_verdad_sintetica.csv")
    salida = RAIZ / "salida"
    salida.mkdir(exist_ok=True)

    resumen = pd.DataFrame([
        dict(perfil=p["id"], nombre=p["nombre"], personas=p["n"], porcentaje_poblacion=round(p["share"] * 100, 1),
             participacion_2018=round(p["v2018"] * 100, 1), participacion_2021=round(p["v2021"] * 100, 1),
             participacion_2024=round(p["v2024"] * 100, 1), edad_media=round(p["edad"], 1),
             escolaridad_media=round(p["escolaridad"], 1), ingreso_mediano_usd=round(p["ingreso"]),
             barreras="; ".join(p["barreras"]),
             estrategias="; ".join(f"{e['nombre']} (+{e['pp']:.1f} pp)" for e in p["estrategias"]))
        for p in R["perfiles"]])
    resumen.to_csv(salida / "perfiles_resumen.csv", index=False, encoding="utf-8-sig")
    cols = ["id", "perfil", "prob_participacion", "voto_2018", "voto_2021", "voto_2024"]
    R["df"][cols].to_csv(salida / "ciudadanos_con_perfil.csv", index=False)

    print("3/3 Construyendo tablero...")
    reporte.construir(R, salida / "reporte.html")

    # Carpeta lista para publicar (Vercel u otro hosting estático)
    publico = RAIZ / "public"
    publico.mkdir(exist_ok=True)
    (publico / "index.html").write_bytes((salida / "reporte.html").read_bytes())
    (publico / "vercel.json").write_text(
        '{\n  "cleanUrls": true,\n  "headers": [\n    {\n      "source": "/(.*)",\n'
        '      "headers": [\n        { "key": "X-Content-Type-Options", "value": "nosniff" },\n'
        '        { "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }\n      ]\n    }\n  ]\n}\n',
        encoding="utf-8")

    s, e = R["seg"], R["ev"]
    print(f"\nk = {s['k']} · estabilidad ARI = {s['estabilidad']:.2f} · AUC = {e['auc']:.3f}")
    print(resumen[["perfil", "nombre", "personas", "participacion_2024"]].to_string(index=False))
    print(f"\nListo: {salida / 'reporte.html'}")


if __name__ == "__main__":
    main()
