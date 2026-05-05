# Entrenamiento — Xrays Evaluation

Este documento explica cómo ejecutar y replicar los experimentos de entrenamiento del servicio `xrays_evaluation`.

El objetivo es entrenar modelos YOLO de clasificación para radiografías de tórax, registrando cada experimento en MLflow para comparar resultados entre arquitecturas.

---

## 1. Servicio

```text
python/xrays_evaluation
```

Tarea:

```text
image_classification
```

Modelos:

```text
YOLO11n-cls
YOLO11m-cls
YOLO11x-cls
```

Experimento MLflow:

```text
ingenia_services/xrays_evaluation/training
```

> Nota: si en tu repositorio ya estás usando el nombre `ingeniia_services/xrays_evaluation/training`, conserva ese nombre en los YAML y en MLflow.

---

## 2. Estructura relevante

```text
python/xrays_evaluation/
│
├── config/
│   └── training/
│       └── experiments/
│           ├── 01-xrays_evaluation-yolo11n_cls-xrays_evaluation_img-v100-training.yaml
│           ├── 02-xrays_evaluation-yolo11m_cls-xrays_evaluation_img-v100-training.yaml
│           └── 03-xrays_evaluation-yolo11x_cls-xrays_evaluation_img-v100-training.yaml
│
├── reports/
│
└── src/
    ├── process_data/
    │   └── collect_data/
    │
    └── training/
        ├── train.py
        └── pretrained_models/
            ├── yolo11n-cls.pt
            ├── yolo11m-cls.pt
            └── yolo11x-cls.pt
```

---

## 3. MLflow centralizado

Este servicio no debe crear un `mlflow.db` local.

Todos los runs deben enviarse a:

```text
http://127.0.0.1:5000
```

El YAML debe contener:

```yaml
mlops_config:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "ingeniia_services/xrays_evaluation/training"
```

---

## 4. Levantar MLflow Server

Desde la raíz del repositorio:

```powershell
cd C:\Users\santi\Desktop\proyectos\ingeniia_services

.\.venv\Scripts\Activate.ps1

python -m mlflow server `
  --backend-store-uri sqlite:///.mlflow/mlflow.db `
  --default-artifact-root file:///C:/Users/santi/Desktop/proyectos/ingeniia_services/.mlflow/artifacts `
  --host 127.0.0.1 `
  --port 5000
```

Abrir:

```text
http://127.0.0.1:5000
```

---

## 5. Dataset

Dataset esperado:

```text
datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121/split_data
```

Estructura esperada por Ultralytics Classification:

```text
split_data/
│
├── train/
│   ├── class_0/
│   └── class_1/
│
├── valid/
│   ├── class_0/
│   └── class_1/
│
└── test/
    ├── class_0/
    └── class_1/
```

La ruta del dataset se define en cada YAML:

```yaml
data_source:
  dataset_path: "../../datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121/split_data"
```

Las rutas relativas se resuelven desde:

```text
python/xrays_evaluation
```

---

## 6. Ejecutar entrenamiento

Desde:

```powershell
cd C:\Users\santi\Desktop\proyectos\ingeniia_services\python\xrays_evaluation

.\.venv\Scripts\Activate.ps1
```

Entrenar YOLO11n-cls:

```powershell
python src/training/train.py --config config/training/experiments/01-xrays_evaluation-yolo11n_cls-xrays_evaluation_img-v100-training.yaml
```

Entrenar YOLO11m-cls:

```powershell
python src/training/train.py --config config/training/experiments/02-xrays_evaluation-yolo11m_cls-xrays_evaluation_img-v100-training.yaml
```

Entrenar YOLO11x-cls:

```powershell
python src/training/train.py --config config/training/experiments/03-xrays_evaluation-yolo11x_cls-xrays_evaluation_img-v100-training.yaml
```

---

## 7. Salidas locales

Los resultados locales se guardan en:

```text
python/xrays_evaluation/runs/
```

O según el YAML:

```yaml
local_runs_dir: "runs/YOLO11_CLS"
```

Los reportes auxiliares se guardan en:

```text
python/xrays_evaluation/reports/
```

---

## 8. Artifacts registrados en MLflow

El entrenamiento registra:

```text
configs/
dataset/
weights/
```

También registra métricas finales expuestas por Ultralytics, por ejemplo:

```text
final_metrics/accuracy_top1
final_metrics/accuracy_top5
dataset_train_images_total
dataset_valid_images_total
dataset_test_images_total
```

---

## 9. Comparación de modelos

Comparar principalmente:

```text
top1_acc
top5_acc
tiempo de inferencia
tamaño del modelo
```

Criterio práctico:

```text
YOLO11n-cls → modelo ligero
YOLO11m-cls → balance
YOLO11x-cls → mayor capacidad
```

---

## 10. Buenas prácticas

No modificar `train.py` para cambiar modelos o hiperparámetros.

Crear o editar YAMLs en:

```text
config/training/experiments/
```

Cada run debe tener un nombre único.

Ejemplo:

```text
EXP_004-xrays_evaluation-YOLO11sCLS-xrays_evaluation_img_v110-training
```

---

## 11. Resumen

Este servicio usa el estándar:

```text
YAML + MLflow central + pesos preentrenados + dataset summary + validación final
```

Para repetir un entrenamiento, ejecutar el YAML correspondiente.
