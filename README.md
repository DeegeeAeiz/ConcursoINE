# Perfiles ciudadanos de participación

Identifica patrones de comportamiento electoral, construye perfiles de participación ciudadana y
propone estrategias para incentivar una mayor participación. Todo el flujo está en Python y los datos
son **sintéticos** (se generan en el mismo proyecto).

## Ejecutar

```bash
pip install -r requirements.txt
python main.py          # genera datos, analiza y construye el tablero (≈1 min)
python verificar.py     # 18 verificaciones (integridad de datos y paridad Python <-> JavaScript; requiere Node)
```

Abre `salida/reporte.html` en el navegador (archivo autocontenido, funciona sin conexión).

## Qué contiene

| Archivo | Función |
|---|---|
| `generar_datos.py` | Datos sintéticos: 30,000 personas, 9 grupos latentes, voto en 2018/2021/2024 |
| `analisis.py` | Limpieza, K-Means (k = 9), regresión logística, escenarios de intervención |
| `reporte.py` + `static/` | Tablero interactivo (Plotly + HTML/CSS/JS) |
| `main.py` | Ejecuta todo el flujo |
| `verificar.py` | Pruebas automáticas |
| `salida/` | `reporte.html`, `perfiles_resumen.csv`, `ciudadanos_con_perfil.csv` |

## Método en breve

1. **Perfiles:** K-Means sobre 11 variables estandarizadas + historial de voto; k = 9 fijado por diseño (silhouette reportado como referencia); estabilidad por submuestreo.
2. **Factores:** regresión logística de la participación 2024 (AUC ≈ 0.80, calibrada); importancia por permutación.
3. **Estrategias:** cinco palancas (casilla cercana, contacto personalizado, educación cívica, confianza, vinculación comunitaria) simuladas con el modelo, con cobertura ajustable.
4. **Simulador de ciudadano:** el mismo modelo y centroides corriendo en JavaScript.

Las estimaciones de impacto son asociaciones del modelo, no efectos causales: validarlas con un piloto con grupo de control.
