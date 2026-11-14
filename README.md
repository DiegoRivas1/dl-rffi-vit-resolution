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
| 3b | Controles de MSViT: redimensionamiento, sensibilidad por eje, parche escalado | hecha |

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

### 2. Fase 1 — ViT sin fine-tuning

```powershell
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml --resume      # continuar tras Ctrl+C
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml --overwrite   # empezar de cero
python -m scripts.evaluate --config configs/experiments/f1_vit_base_r64.yaml --ckpt best
```

Semillas 43–46: mismos comandos con `f1_vit_base_r64_s43.yaml` … `_s46.yaml`.

`train` entrena en `R_train` y al terminar evalúa en todas las `R_test` con STFT regenerada. En el ViT, el positional
embedding se interpola (bicúbica) cuando cambia la cuadrícula de parches; no se actualiza ningún peso.

### 3. Control — entrenar y probar en la misma resolución

```powershell
python -m scripts.train --config configs/experiments/f1c_vit_train_r32x32.yaml
python -m scripts.train --config configs/experiments/f1c_vit_train_r96x64.yaml
python -m scripts.train --config configs/experiments/f1c_vit_train_r128x128.yaml
```

Sirve para descartar que la caída de la fase 1 sea por falta de información en la resolución de prueba.

### 4. Fase 2 — ViT con fine-tuning

```powershell
python -m scripts.finetune --config configs/experiments/f2_vit_ft_r32x32.yaml
python -m scripts.finetune --config configs/experiments/f2_vit_ft_r96x64.yaml
python -m scripts.finetune --config configs/experiments/f2_vit_ft_r128x128.yaml
```

Parte de `runs/f1_vit_base_r64/checkpoints/best.pt`, adapta el positional embedding a la resolución objetivo y
reentrena 20 epochs con lr 1e-4. Requiere haber corrido la fase 1. Acepta `--resume` y `--overwrite`.

### 5. Fase 3 — MSViT sin fine-tuning

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

# c) MSViT con parche fijo vs parche escalado con la resolución
python -m scripts.eval_msvit_scale --runs f3_msvit_base_r64 f3_msvit_base_r64_s43 f3_msvit_base_r64_s44 f3_msvit_base_r64_s45 f3_msvit_base_r64_s46
```

En (a) y (b) conviene pasar los 10 runs (5 de ViT y 5 de MSViT) para tener media y desviación.

### 7. Agregado entre semillas y comparación

```powershell
python -m scripts.plot_seeds                    # los 3 modos de interpolación
python -m scripts.plot_seeds --modes bilinear
jupyter lab notebooks/comparison.ipynb
```

`plot_seeds` necesita haber corrido antes `eval_resize` para cada run y modo.

## Resultados (exactitud en prueba, 900 señales, azar = 0.167)

Entrenados en `R64x64`, evaluados sin fine-tuning con STFT regenerada (media ± desv., 5 semillas):

| Prueba | ViT | MSViT, parche fijo | MSViT, parche escalado |
|---|---|---|---|
| `R64x64` | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.999 ± 0.001 |
| `R32x32` | 0.197 ± 0.056 | 0.239 ± 0.068 | 0.354 ± 0.179 |
| `R96x64` | 0.162 ± 0.063 | 0.187 ± 0.092 | 0.458 ± 0.202 |
| `R128x128` | 0.167 ± 0.000 | 0.167 ± 0.000 | 0.317 ± 0.134 |

- **Control (fase 1c):** entrenar y probar en la misma resolución da 0.997–1.000 en las cuatro. La información está
  presente; lo que falla es la transferencia.
- **Fine-tuning (fase 2, semilla 42):** 0.998–0.999 en la resolución objetivo; en `R64x64` quedan 0.72 / 0.54 / 0.81.
- **Redimensionamiento (bilinear) con MSViT:** parche fijo 0.153 / 0.329 / 0.167 y parche escalado 0.488 / 0.998 / 0.998
  en `R32x32` / `R96x64` / `R128x128`.
- **Por eje:** cambiar solo T conserva 0.64–0.83 en el ViT; cambiar F colapsa ambos modelos.

## Notas sobre MSViT

- Es una **adaptación** de Jiang et al., *Extending the scale generalization of the Vision Transformer without
  fine-tuning*, Neural Networks 205 (2027) 109446. No hay código oficial. Diferencias: 1 canal (sin YCbCr), tamaño
  reducido (3 etapas, dims 64/128/256, 3.26 M parámetros), AdamW en lugar de Adan, sin RandAugment/MixUp/CutMix.
- En el paper el parche es **fijo** y la rejilla de tokens crece con la resolución; así se evalúa por defecto.
- El **parche escalado** (`model.ref_hw`) es una variante propia: el parche crece con la resolución y se proyecta a
  los mismos canales DCT, de modo que la rejilla de tokens no cambia. No añade pesos ni requiere reentrenar.
- Al ampliar por redimensionamiento, el parche escalado recupera casi por construcción los tokens originales; el
  resultado informativo es el de STFT regenerada.

## Estructura

```
configs/preprocess/    datos + resoluciones STFT (fuente única)
configs/experiments/   un YAML por experimento (el nombre del modelo = experiment.name)
                       f1_*  ViT base · f1c_*  control por resolución · f2_*  fine-tuning · f3_*  MSViT · _sNN  semilla
preprocessing/         dataset_loader, splits, stft_generator, explore_stft (+ README con lineamientos)
src/models/            vit.py · msvit.py
src/training/          trainer (checkpoints reanudables), data, setup
src/evaluation/        métricas, evaluación por resolución y de fine-tuning
scripts/               preprocess, train, evaluate, finetune, eval_resize, eval_resize_grid, eval_msvit_scale,
                       plot_seeds, inspect_manysig, make_synthetic_pkl
runs/<experimento>/    config_snapshot.yaml, checkpoints/{last,best}.pt, metrics/*.csv, figures/*.png  (no versionado)
figures/               exploration/ y comparison/ (figuras y tablas agregadas)
notebooks/             comparison.ipynb (solo comparaciones)
```

Figuras: `<experimento>__<tipo>__<características>.png`, p. ej. `f1_vit_base_r64__confmat__train-R64x64_test-R96x64.png`.
