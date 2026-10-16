# Lineamientos de preprocesamiento

1. **Fuente:** WiSig `ManySig.pkl` (6 Tx, 12 Rx, 4 fechas, 2 estados `equalized`, 1000 señales de `(256, 2)` = I/Q).
   Acceso oficial: `data["data"][tx][rx][fecha][equalized]`. Referencia: Hanna et al., *WiSig*, IEEE Access 2022 (ref. [4] de la tesis).
2. **Subconjunto controlado:** `equalized`, Rx y fecha fijos en el YAML para aislar el efecto de la resolución
   (se puede ampliar a más Rx/fechas sin cambiar el código). El nº de señales por celda lo fija `n_per_cell`.
3. **Particiones primero:** `splits.py` divide por señal (estratificada por Tx) ANTES de generar ninguna resolución.
   Si una señal está en test, todas sus resoluciones están en test. Nunca split independiente por resolución.
4. **Resolución = parámetros de generación STFT** (`n_fft`, `win_len`, `hop`), no un resize de una imagen ya generada.
   `F = n_fft`, `T = 256 // hop`. Usar `win_len = n_fft` evita bins interpolados por zero-padding.
5. **Normalización por muestra** (log-potencia en dB + z-score de cada espectrograma): no usa estadísticas del conjunto de prueba
   ni depende de la resolución.
6. **Compatibilidad con ViT:** `F` y `T` deben ser múltiplos del tamaño de parche.
7. **Reproducibilidad:** semilla del split en el YAML; el subset y los splits se guardan en `data/processed/`.
8. El YAML `configs/preprocess/*.yaml` es la fuente única de verdad de datos + resoluciones.
