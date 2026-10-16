# dl-rffi-vit-resolution

Experimentación de la tesis *Generalización de Vision Transformers ante cambios de resolución para la
identificación de dispositivos mediante RF Fingerprinting* (UNSA, EPCC).

| Fase | Contenido | Estado |
|---|---|---|
| 1 | ViT convencional, entrenado en `R_train`, evaluado SIN fine-tuning en `R_test` | **implementada** |
| 2 | ViT + fine-tuning en `R_test` (control de adaptación; se reporta aparte) | pendiente |
| 3 | MSViT (SC-Conv, HVSA, RepCPE; Jiang et al.) sin fine-tuning | pendiente |

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

## Flujo de trabajo

```powershell
# 0) Pruebas aisladas (sin datos)
python -m src.utils.naming
python -m preprocessing.stft_generator
python -m preprocessing.dataset_loader
python -m preprocessing.splits
python -m src.models.vit

# 1) Explorar resoluciones posibles (tabla + espectrogramas) y definirlas en el YAML de preprocesamiento
python -m preprocessing.explore_stft
python -m preprocessing.explore_stft --pkl data/raw/ManySig.pkl      # chequeo equalized 0 vs 1

# 2) Subconjunto + particiones (mover ManySig.pkl a data/raw/ o usar --pkl)
python -m scripts.preprocess --config configs/preprocess/manysig_eq1.yaml

# 3) Entrenar + evaluar en todas las resoluciones (sin fine-tuning)
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml
#    Pausar: Ctrl+C   ·  Continuar:
python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml --resume
#    Empezar de cero:  --overwrite

# 4) Re-evaluar un modelo guardado
python -m scripts.evaluate --config configs/experiments/f1_vit_base_r64.yaml --ckpt best

# 5) Comparar modelos
jupyter lab notebooks/comparison.ipynb

# Prueba rápida del pipeline con datos sintéticos (sin el pkl de 2.3 GB)
python -m scripts.make_synthetic_pkl
```

## Estructura

```
configs/preprocess/    datos + resoluciones STFT (fuente única)
configs/experiments/   un YAML por experimento (el nombre del modelo = experiment.name)
preprocessing/         dataset_loader, splits, stft_generator, explore_stft (+ README con lineamientos)
src/models/            vit.py (fase 1) · aquí irá msvit.py (fase 3)
src/training/          trainer (checkpoints reanudables), data, setup
src/evaluation/        métricas y evaluación por resolución
scripts/               preprocess, train, evaluate, inspect_manysig, make_synthetic_pkl
runs/<experimento>/    config_snapshot.yaml, checkpoints/{last,best}.pt, metrics/*.csv, figures/*.png
notebooks/             comparison.ipynb (solo comparaciones)
```

Figuras: `<experimento>__<tipo>__<características>.png`, p. ej. `f1_vit_base_r64__confmat__train-R64x64_test-R96x64.png`.
