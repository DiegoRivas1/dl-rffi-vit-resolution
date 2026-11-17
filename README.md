# dl-rffi-vit-resolution

Experimentación de la tesis *Generalización de Vision Transformers ante cambios de resolución para la
identificación de dispositivos mediante RF Fingerprinting* (UNSA, EPCC).

Pregunta: un ViT entrenado con espectrogramas de una resolución (`R64x64`), ¿sigue identificando al transmisor
cuando la resolución de prueba cambia, **sin fine-tuning**? ¿Lo mejora una arquitectura multiescala (MSViT)?

| Fase | Contenido | Estado |
|---|---|---|
| 1 | ViT convencional, entrenado en `R_train`, evaluado sin fine-tuning en `R_test` | hecha (5 semillas) |
| 1c | Control: ViT entrenado y probado en la misma resolución | hecha |
| 2 | ViT + fine-tuning en `R_test` (cota de adaptación; se reporta aparte) | hecha (1 semilla) |
| 3 | MSViT adaptado (SC-Conv, HVSA, RepCPE; Jiang et al.) sin fine-tuning | hecha (5 semillas) |
| 3b | Controles: redimensionamiento, sensibilidad por eje, parche escalado, ventana fija, barrido de ventana | hecha |
| 4 | Robustez: otros receptores y fechas; réplica con 12 receptores y fecha de prueba no vista | hecha (3 semillas) |

## Datos y resoluciones

WiSig `ManySig.pkl`: 6 transmisores (clases), `equalized = 1`, receptor `1-1`, fecha `2021_03_01` → 6000 señales I/Q
de 256 muestras. Partición estratificada 70/15/15 (4200/900/900, semilla 42), hecha **antes** de generar resoluciones.

La resolución se define por los parámetros de la STFT (no por resize): `F = n_fft`, `T = 256 // hop`, `win_len = n_fft`.

| Nombre | n_fft | hop | Forma F×T | Uso |
|---|---|---|---|---|
| `R32x32` | 32 | 8 | 32×32 | prueba |
| `R64x64` | 64 | 4 | 64×64 | **entrenamiento** |
| `R96x64` | 96 | 4 | 96×64 | prueba |
| `R128x128` | 128 | 2 | 128×128 | prueba |

Todo se configura en `configs/preprocess/manysig_eq1.yaml` (ver `preprocessing/README.md`).

## Instalación (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# PyTorch con CUDA (ajustar cu128 si corresponde)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python -m ipykernel install --user --name dl-rffi-vit --display-name "Python (dl-rffi-vit)"
```

## Comandos por sección

Todos se ejecutan desde la raíz del repositorio.

### 0. Pruebas aisladas (sin datos)

```powershell
python -m src.utils.naming
python -m preprocessing.stft_generator      # formas y posición de un tono por resolución
python -m preprocessing.dataset_loader
python -m preprocessing.splits
python -m src.models.vit                    # incluye la interpolación del positional embedding
python src/models/msvit.py                  # DCT, SC-Conv fusionada, Toeplitz, HVSA, parche escalado
python src/models/patch_scale.py            # reescalado de un kernel de parche en el dominio DCT (ViT)
```

### 1. Preprocesamiento

```powershell
python -m scripts.inspect_manysig --pkl data/raw/ManySig.pkl         # estructura del pkl
python -m preprocessing.explore_stft                                 # tabla de resoluciones posibles
python -m preprocessing.explore_stft --pkl data/raw/ManySig.pkl      # + espectrogramas reales (equalized 0 vs 1)
python -m scripts.preprocess --config configs/preprocess/manysig_eq1.yaml
```

Salidas: `data/processed/<nombre>_subset.npz` y `_splits.npz`; figura
`figures/exploration/explore__spectrogram_grid__data-real.png`.

Sin el pkl de 2.3 GB se puede probar el pipeline con `python -m scripts.make_synthetic_pkl`.

### 2. Fase 1 : ViT sin fine-tuning

```powershell
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml --resume      # continuar tras Ctrl+C
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml --overwrite   # empezar de cero
python -m scripts.evaluate --config configs/experiments/f1_vit_base_r64.yaml --ckpt best
```

Semillas 43–46: mismos comandos con `f1_vit_base_r64_s43.yaml` … `_s46.yaml`.

`train` entrena en `R_train` y al terminar evalúa en todas las `R_test` con STFT regenerada. En el ViT, el positional
embedding se interpola (bicúbica) cuando cambia la cuadrícula de parches; no se actualiza ningún peso.

### 3. Control : entrenar y probar en la misma resolución

```powershell
python -m scripts.train --config configs/experiments/f1c_vit_train_r32x32.yaml
python -m scripts.train --config configs/experiments/f1c_vit_train_r96x64.yaml
python -m scripts.train --config configs/experiments/f1c_vit_train_r128x128.yaml
```

Sirve para descartar que la caída de la fase 1 sea por falta de información en la resolución de prueba.

### 4. Fase 2 : ViT con fine-tuning

```powershell
python -m scripts.finetune --config configs/experiments/f2_vit_ft_r32x32.yaml
python -m scripts.finetune --config configs/experiments/f2_vit_ft_r96x64.yaml
python -m scripts.finetune --config configs/experiments/f2_vit_ft_r128x128.yaml
```

Parte de `runs/f1_vit_base_r64/checkpoints/best.pt`, adapta el positional embedding a la resolución objetivo y
reentrena 20 epochs con lr 1e-4. Requiere haber corrido la fase 1. Acepta `--resume` y `--overwrite`.

### 5. Fase 3 : MSViT sin fine-tuning

```powershell
python -m scripts.train --config configs/experiments/f3_msvit_base_r64.yaml
```

Semillas 43–46: `f3_msvit_base_r64_s43.yaml` … `_s46.yaml`. Mismo protocolo, particiones y receta que la fase 1.

### 6. Controles sin reentrenar (usan los `best.pt` existentes)

```powershell
# a) Redimensionar el espectrograma de R_train en lugar de regenerar la STFT
python -m scripts.eval_resize --runs f1_vit_base_r64 f3_msvit_base_r64
python -m scripts.eval_resize --runs f1_vit_base_r64 f3_msvit_base_r64 --mode bicubic    # bilinear | bicubic | nearest

# b) Sensibilidad por eje: redimensionar a (F, T) arbitrarios
python -m scripts.eval_resize_grid --runs f1_vit_base_r64 f1_vit_base_r64_s43 f3_msvit_base_r64 f3_msvit_base_r64_s43
python -m scripts.eval_resize_grid --runs ... --sizes 32 48 64 80 96 128

# c) Parche fijo vs parche escalado con la resolución, en ViT y MSViT, con tres tipos de entrada:
#    STFT regenerada (ventana = n_fft), STFT de ventana fija (resolutions_fixed_window del YAML) y redimensionada
python -m scripts.eval_scaled_patch --runs f1_vit_base_r64 f1_vit_base_r64_s43 f1_vit_base_r64_s44 f1_vit_base_r64_s45 f1_vit_base_r64_s46 f3_msvit_base_r64 f3_msvit_base_r64_s43 f3_msvit_base_r64_s44 f3_msvit_base_r64_s45 f3_msvit_base_r64_s46
```

```powershell
# d) Barrido de ventana: imagen siempre 64x64 (mismo n_fft y hop); solo cambia la longitud de la ventana STFT
python -m scripts.eval_window_sweep --runs f1_vit_base_r64 f1_vit_base_r64_s43 f1_vit_base_r64_s44 f1_vit_base_r64_s45 f1_vit_base_r64_s46 f3_msvit_base_r64 f3_msvit_base_r64_s43 f3_msvit_base_r64_s44 f3_msvit_base_r64_s45 f3_msvit_base_r64_s46
```

Salida de (d): `figures/comparison/window_sweep__acc_vs_window__n_runs-10.png` y su tabla `window_sweep__summary__n_runs-10.csv`.

Salida de (c): `runs/<run>/metrics/eval_scaled_patch_bilinear.csv` y la tabla agregada
`figures/comparison/scaled_patch__summary__mode-bilinear_n_runs-10.csv`.

En (a) y (b) conviene pasar los 10 runs (5 de ViT y 5 de MSViT) para tener media y desviación.

### 7. Agregado entre semillas y comparación

```powershell
python -m scripts.plot_seeds                    # los 3 modos de interpolación
python -m scripts.plot_seeds --modes bilinear
jupyter lab notebooks/comparison.ipynb
jupyter lab notebooks/visualizacion_entradas.ipynb
```

`visualizacion_entradas.ipynb` muestra qué recibe el modelo en cada caso (señal I/Q, espectrograma, huella por
transmisor, los tres niveles de cambio de resolución, barrido de ventana, parche fijo y escalado) y cómo responde
(tokens y predicciones). Usa los `best.pt` de `f1_vit_base_r64` y `f3_msvit_base_r64` y guarda las figuras en
`figures/notebook/`.

`plot_seeds` necesita haber corrido antes `eval_resize` para cada run y modo.

### 8. Fase 4 : robustez con más receptores y fechas

```powershell
# a) Sin reentrenar: los modelos de 1 receptor / 1 fecha, probados en otras fechas y otros receptores (en R64x64)
python -m scripts.eval_cross_condition --runs f1_vit_base_r64 f1_vit_base_r64_s43 f1_vit_base_r64_s44 f1_vit_base_r64_s45 f1_vit_base_r64_s46 f3_msvit_base_r64 f3_msvit_base_r64_s43 f3_msvit_base_r64_s44 f3_msvit_base_r64_s45 f3_msvit_base_r64_s46

# b) Réplica: 12 receptores, entrenamiento en 3 fechas y prueba en una fecha no vista
python -m scripts.preprocess --config configs/preprocess/manysig_eq1_multi.yaml
python -m scripts.train --config configs/experiments/f4_vit_multi_r64.yaml        # y _s43, _s44
python -m scripts.train --config configs/experiments/f4_msvit_multi_r64.yaml      # y _s43, _s44

# c) Mismos controles que en la sección 6, sobre los modelos de la fase 4
python -m scripts.eval_scaled_patch --runs f4_vit_multi_r64 f4_vit_multi_r64_s43 f4_vit_multi_r64_s44 f4_msvit_multi_r64 f4_msvit_multi_r64_s43 f4_msvit_multi_r64_s44
python -m scripts.eval_window_sweep --runs f4_vit_multi_r64 f4_vit_multi_r64_s43 f4_vit_multi_r64_s44 f4_msvit_multi_r64 f4_msvit_multi_r64_s43 f4_msvit_multi_r64_s44
```

(a) mide cuánto del ~100 % depende del receptor y del día de entrenamiento. (b) y (c) repiten el estudio de
resolución en un escenario donde la prueba es un día que el modelo nunca vio (72 000 señales: 6 Tx × 12 Rx × 4 fechas
× 250; la última fecha completa es la prueba).

## Resultados (exactitud en prueba, 900 señales, azar = 0.167)

Modelos entrenados en `R64x64` y evaluados sin fine-tuning (media ± desv., 5 semillas por modelo). En `R64x64` todos
obtienen 0.999 ± 0.001. Tabla completa: `figures/comparison/scaled_patch__summary__mode-bilinear_n_runs-10.csv`.

| Entrada | Prueba | ViT, parche fijo | ViT, parche escalado | MSViT, parche fijo | MSViT, parche escalado |
|---|---|---|---|---|---|
| Redimensionada (bilinear) | `R32x32` | 0.143 ± 0.068 | 0.390 ± 0.125 | 0.153 ± 0.056 | 0.488 ± 0.303 |
| | `R96x64` | 0.159 ± 0.057 | 0.998 ± 0.001 | 0.329 ± 0.164 | 0.998 ± 0.001 |
| | `R128x128` | 0.162 ± 0.010 | 0.998 ± 0.001 | 0.167 ± 0.000 | 0.998 ± 0.001 |
| STFT de ventana fija (64) | `R32x32` | 0.143 ± 0.053 | 0.234 ± 0.065 | 0.216 ± 0.070 | 0.497 ± 0.268 |
| | `R96x64` | 0.118 ± 0.041 | 0.959 ± 0.072 | 0.192 ± 0.025 | 0.974 ± 0.037 |
| | `R128x64` (solo frecuencia) | 0.167 ± 0.000 | 0.888 ± 0.120 | 0.167 ± 0.000 | 0.901 ± 0.093 |
| | `R64x128` (solo tiempo) | 0.677 ± 0.064 | 0.998 ± 0.002 | 0.573 ± 0.052 | 0.999 ± 0.001 |
| | `R128x128` | 0.167 ± 0.000 | 0.880 ± 0.141 | 0.167 ± 0.000 | 0.892 ± 0.117 |
| STFT regenerada (ventana = n_fft) | `R32x32` | 0.197 ± 0.056 | 0.432 ± 0.119 | 0.239 ± 0.068 | 0.354 ± 0.179 |
| | `R96x64` | 0.162 ± 0.063 | 0.498 ± 0.115 | 0.187 ± 0.092 | 0.458 ± 0.202 |
| | `R128x128` | 0.167 ± 0.000 | 0.327 ± 0.101 | 0.167 ± 0.000 | 0.317 ± 0.134 |

Lectura:

- **Con parche fijo ambos modelos colapsan** fuera de `R64x64`, sea cual sea el tipo de entrada. La única excepción
  parcial es cambiar solo el eje de tiempo (0.57–0.68).
- **El parche escalado explica la mejora, no la arquitectura:** ViT y MSViT quedan prácticamente iguales cuando ambos
  lo usan.
- **Tres niveles de cambio de resolución** (al ampliar, con parche escalado): geométrico 0.998; mismo análisis con
  otro muestreo (ventana fija) 0.88–0.97; otro compromiso tiempo-frecuencia (ventana = n_fft) 0.32–0.50. Lo que no se
  transfiere es el cambio de ventana, no el número de píxeles.
- **Reducir a `R32x32` falla en los tres niveles** (0.23–0.50).
- **Control (fase 1c):** entrenar y probar en la misma resolución da 0.997–1.000 en las cuatro.
- **Fine-tuning (fase 2, semilla 42):** 0.998–0.999 en la resolución objetivo; en `R64x64` quedan 0.72 / 0.54 / 0.81.

### Fase 4 — robustez

**a) Modelos de 1 receptor / 1 fecha probados en otras condiciones** (en `R64x64`, 5 semillas por modelo):

| Condición de prueba | ViT | MSViT |
|---|---|---|
| Celda de entrenamiento (señales de prueba) | 0.999 ± 0.001 | 0.999 ± 0.001 |
| Mismo receptor, otras 3 fechas | 0.518 ± 0.035 | 0.514 ± 0.042 |
| Otros 11 receptores, misma fecha | 0.275 ± 0.018 | 0.277 ± 0.024 |

El ~100 % de las fases 1–3 es específico del receptor y del día de entrenamiento.

**b) Réplica con 12 receptores, 3 fechas de entrenamiento y una fecha de prueba no vista** (3 semillas por modelo;
validación 0.99, prueba en `R64x64`: ViT 0.906 ± 0.012, MSViT 0.904 ± 0.010). Tabla completa:
`figures/comparison/scaled_patch__summary__mode-bilinear_n_runs-6.csv`.

| Entrada | Prueba | ViT, parche fijo | ViT, parche escalado | MSViT, parche fijo | MSViT, parche escalado |
|---|---|---|---|---|---|
| Redimensionada (bilinear) | `R32x32` | 0.167 ± 0.021 | 0.299 ± 0.032 | 0.182 ± 0.044 | 0.422 ± 0.046 |
| | `R96x64` | 0.163 ± 0.021 | 0.741 ± 0.020 | 0.232 ± 0.055 | 0.863 ± 0.005 |
| | `R128x128` | 0.166 ± 0.009 | 0.768 ± 0.027 | 0.178 ± 0.050 | 0.862 ± 0.011 |
| STFT de ventana fija (64) | `R32x32` | 0.146 ± 0.041 | 0.248 ± 0.055 | 0.201 ± 0.035 | 0.536 ± 0.021 |
| | `R96x64` | 0.172 ± 0.023 | 0.378 ± 0.064 | 0.225 ± 0.053 | 0.438 ± 0.002 |
| | `R128x64` (solo frecuencia) | 0.173 ± 0.009 | 0.337 ± 0.043 | 0.199 ± 0.058 | 0.361 ± 0.008 |
| | `R64x128` (solo tiempo) | 0.358 ± 0.031 | 0.902 ± 0.013 | 0.468 ± 0.032 | 0.890 ± 0.018 |
| | `R128x128` | 0.144 ± 0.014 | 0.342 ± 0.032 | 0.151 ± 0.048 | 0.353 ± 0.004 |
| STFT regenerada (ventana = n_fft) | `R32x32` | 0.176 ± 0.011 | 0.204 ± 0.007 | 0.227 ± 0.061 | 0.303 ± 0.028 |
| | `R96x64` | 0.210 ± 0.038 | 0.282 ± 0.042 | 0.250 ± 0.038 | 0.383 ± 0.044 |
| | `R128x128` | 0.190 ± 0.039 | 0.253 ± 0.040 | 0.206 ± 0.055 | 0.249 ± 0.043 |

Barrido de ventana a 64×64 (`window_sweep__summary__n_runs-6.csv`): ventana 48 → ViT 0.547 / MSViT 0.804;
56 → 0.789 / 0.861; 64 → 0.906 / 0.904; 72 → 0.859 / 0.886; 80 → 0.567 / 0.713; 96 → 0.330 / 0.472; 128 → 0.236 / 0.356.

Qué se mantiene y qué cambia respecto al escenario de 1 receptor / 1 fecha:

- **Se mantiene:** con parche fijo ambos modelos colapsan al cambiar la resolución; el parche escalado recupera el
  cambio geométrico y el cambio en el eje de tiempo; el cambio de ventana sigue sin resolverse.
- **Cambia:** el remuestreo en frecuencia con ventana fija ya no se recupera (0.34–0.44 frente a 0.88–0.97), y el
  MSViT pasa a ser más robusto que el ViT (redimensionamiento con parche escalado 0.86 frente a 0.74–0.77; mayor
  tolerancia a la ventana en todo el barrido).

## Notas sobre MSViT

- Es una **adaptación** de Jiang et al., *Extending the scale generalization of the Vision Transformer without
  fine-tuning*, Neural Networks 205 (2027) 109446. No hay código oficial. Diferencias: 1 canal (sin YCbCr), tamaño
  reducido (3 etapas, dims 64/128/256, 3.26 M parámetros), AdamW en lugar de Adan, sin RandAugment/MixUp/CutMix.
- En el paper el parche es **fijo** y la rejilla de tokens crece con la resolución; así se evalúa por defecto.
- El **parche escalado** (`model.ref_hw`) es una variante propia: el parche crece con la resolución y se proyecta a
  los mismos canales DCT, de modo que la rejilla de tokens no cambia. No añade pesos ni requiere reentrenar.
- El mismo parche escalado existe para el ViT (`src/models/patch_scale.py`), para que la comparación sea justa.
- **Ventana fija** (`resolutions_fixed_window`): control en el que la STFT conserva la ventana de 64 muestras y solo
  cambia el muestreo en frecuencia (`n_fft`) y en tiempo (`hop`). Separa "cambio de muestreo" de "cambio de ventana".
- Al ampliar por redimensionamiento, el parche escalado recupera casi por construcción los tokens originales; el
  resultado informativo es el de STFT regenerada.

## Estructura

```
configs/preprocess/    datos + resoluciones STFT (fuente única)
configs/experiments/   un YAML por experimento (el nombre del modelo = experiment.name)
                       f1_*  ViT base · f1c_*  control por resolución · f2_*  fine-tuning · f3_*  MSViT · f4_*  robustez · _sNN  semilla
preprocessing/         dataset_loader, splits, stft_generator, explore_stft (+ README con lineamientos)
src/models/            vit.py · msvit.py · patch_scale.py (parche escalado del ViT)
src/training/          trainer (checkpoints reanudables), data, setup
src/evaluation/        métricas, evaluación por resolución y de fine-tuning
scripts/               preprocess, train, evaluate, finetune, eval_resize, eval_resize_grid, eval_scaled_patch, eval_window_sweep, eval_cross_condition,
                       plot_seeds, inspect_manysig, make_synthetic_pkl
runs/<experimento>/    config_snapshot.yaml, metrics/*.csv, figures/*.png  (versionados) · checkpoints/{last,best}.pt (no)
figures/               exploration/ y comparison/ (figuras y tablas agregadas)
notebooks/             comparison.ipynb (comparaciones) · visualizacion_entradas.ipynb (entradas y respuesta del modelo)
```

Figuras: `<experimento>__<tipo>__<características>.png`, p. ej. `f1_vit_base_r64__confmat__train-R64x64_test-R96x64.png`.
