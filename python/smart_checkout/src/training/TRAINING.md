# Entrenamiento — Smart Checkout

Este documento explica cómo ejecutar y replicar los experimentos de entrenamiento del servicio `smart_checkout`.

El objetivo del servicio es entrenar modelos de detección de objetos para identificar productos en un sistema de cajero automático inteligente.

---

## 1. Servicio

```text
python/smart_checkout
```

Tarea:

```text
object_detection
```

Modelo inicial:

```text
YOLO26sDetect
```

Experimento MLflow:

```text
ingenia_services/smart_checkout/training
```

> Nota: si en tu repositorio ya estás usando el nombre `ingeniia_services/smart_checkout/training`, conserva ese nombre en los YAML y en MLflow.

---

## 2. Estructura relevante

```text
python/smart_checkout/
│
├── config/
│   ├── service/
│   │   ├── products.yaml
│   │   └── settings.py
│   │
│   └── training/
│       └── experiments/
│           └── 01-smart_checkout-yolo26s_detect-smart_checkout_img-v100-training.yaml
│
├── reports/
│
└── src/
    ├── process_data/
    │
    └── training/
        └── train.py
```

---

## 3. MLflow centralizado

Todos los experimentos se registran en el servidor central:

```text
http://127.0.0.1:5000
```

El YAML debe contener:

```yaml
mlops_config:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "ingeniia_services/smart_checkout/training"
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
datasets/ingeniia_services_smart_checkout_img_v1.0.0_training_20260120/split_data/data_augmentation/data.yaml
```

Formato esperado:

```text
YOLO Detect
```

Estructura típica:

```text
data_augmentation/
│
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

Cada label debe tener formato:

```text
class_id x_center y_center width height
```

Todas las coordenadas deben estar normalizadas entre `0` y `1`.

---

## 6. Archivo de configuración

Archivo principal:

```text
config/training/experiments/01-smart_checkout-yolo26s_detect-smart_checkout_img-v100-training.yaml
```

Puntos clave:

```yaml
model_config:
  base_model: "yolo26s.pt"
  model_family: "YOLO26sDetect"
  task: "detect"

mlops_config:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "ingeniia_services/smart_checkout/training"
  run_name: "EXP_001-smart_checkout-YOLO26sDetect-smart_checkout_img_v100-training"
  local_runs_dir: "reports/runs/YOLO/yolo26s-detect"
```

---

## 7. Ejecutar entrenamiento

Desde:

```powershell
cd C:\Users\santi\Desktop\proyectos\ingeniia_services\python\smart_checkout

.\.venv\Scripts\Activate.ps1
```

Ejecutar:

```powershell
python src/training/train.py --config config/training/experiments/01-smart_checkout-yolo26s_detect-smart_checkout_img-v100-training.yaml
```

---

## 8. Salidas locales

Los resultados locales se guardan en:

```text
python/smart_checkout/reports/runs/YOLO/yolo26s-detect/
```

Dentro se genera una carpeta con el nombre del run:

```text
EXP_001-smart_checkout-YOLO26sDetect-smart_checkout_img_v100-training
```

---

## 9. Artifacts registrados en MLflow

El pipeline registra:

```text
configs/
dataset/
weights/
```

También registra métricas como:

```text
dataset_train_images_total
dataset_train_instances_total
dataset_val_images_total
dataset_val_instances_total
final_mAP50
final_mAP50-95
```

Los nombres exactos pueden variar según la salida de Ultralytics.

---

## 10. Qué revisar después de entrenar

En MLflow abrir:

```text
ingenia_services/smart_checkout/training
```

O, si el experimento fue creado con el nombre usado en los YAML recientes:

```text
ingenia_services/smart_checkout/training
```

Revisar:

```text
mAP50
mAP50-95
precision
recall
box_loss
cls_loss
dfl_loss
```

También revisar artifacts:

```text
configs/
dataset/
weights/best.pt
weights/last.pt
```

---

## 11. Buenas prácticas

No editar `train.py` para cambiar hiperparámetros.

Crear nuevos YAMLs para cada experimento.

Ejemplo:

```text
02-smart_checkout-yolo26n_detect-smart_checkout_img-v100-training.yaml
03-smart_checkout-yolo26m_detect-smart_checkout_img-v100-training.yaml
```

Posible comparación futura:

```text
YOLO26nDetect → velocidad
YOLO26sDetect → balance
YOLO26mDetect → mayor precisión
```

---

## 12. Resumen

Este servicio sigue el estándar:

```text
YAML + MLflow central + dataset summary + validación final + artifacts
```

El entrenamiento deja de ser una corrida aislada y se convierte en un experimento reproducible y comparable.
