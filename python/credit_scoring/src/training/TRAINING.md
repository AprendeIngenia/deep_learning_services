# Entrenamiento — Credit Scoring

Este documento explica cómo ejecutar y replicar los experimentos de entrenamiento del servicio `credit_scoring`.

El objetivo de este estándar es que cualquier persona pueda reproducir un entrenamiento, revisar sus métricas en MLflow, comparar diferentes arquitecturas y localizar los artefactos generados: configuración, modelo, preprocesador, reportes y métricas.

---

## 1. Servicio

```text
python/credit_scoring
```

Este servicio entrena modelos MLP para clasificación binaria de riesgo crediticio.

Tarea:

```text
binary_classification
```

Modelo:

```text
MLP
```

Experimento MLflow:

```text
ingenia_services/credit_scoring/training
```

> Nota: si en tu repositorio ya estás usando el nombre `ingeniia_services/credit_scoring/training`, conserva ese nombre en los YAML y en MLflow.

---

## 2. Estructura relevante

```text
python/credit_scoring/
│
├── config/
│   └── training/
│       └── experiments/
│           ├── 01-credit_scoring-mlp-german_credit_risk-v100-training.yaml
│           ├── 02-credit_scoring-mlp-german_credit_risk-v100-training.yaml
│           ├── 03-credit_scoring-mlp-german_credit_risk-v100-training.yaml
│           └── 04-credit_scoring-mlp-german_credit_risk-v100-training.yaml
│
├── models/
│
├── reports/
│
└── src/
    ├── processing/
    └── training/
        └── train.py
```

---

## 3. MLflow centralizado

Este proyecto usa un servidor central de MLflow levantado desde la raíz del repositorio.

No se debe usar una base SQLite local dentro de cada servicio.

Correcto:

```text
http://127.0.0.1:5000
```

Incorrecto:

```text
sqlite:///mlflow.db
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

Abrir en navegador:

```text
http://127.0.0.1:5000
```

---

## 5. Verificar conexión con MLflow

Desde el entorno del servicio:

```powershell
cd C:\Users\santi\Desktop\proyectos\ingeniia_services\python\credit_scoring

.\.venv\Scripts\Activate.ps1

python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

Debe mostrar:

```text
http://127.0.0.1:5000
```

---

## 6. Dataset

El dataset se encuentra en:

```text
datasets/ingeniia_services_credit_scoring_csv_german_credit_risk_v1.0.0_training_20250825/german_credit_risk.csv
```

La ruta se define en cada YAML:

```yaml
data_source:
  dataset_name: "german_credit_risk"
  dataset_version: "v1.0.0"
  data_path:
    dataset_path: "../../datasets/ingeniia_services_credit_scoring_csv_german_credit_risk_v1.0.0_training_20250825/german_credit_risk.csv"
```

Las rutas relativas se resuelven desde:

```text
python/credit_scoring
```

---

## 7. Ejecutar entrenamiento

Desde:

```powershell
cd C:\Users\santi\Desktop\proyectos\ingeniia_services\python\credit_scoring

.\.venv\Scripts\Activate.ps1
```

Ejecutar experimento 01:

```powershell
python src/training/train.py --config config/training/experiments/01-credit_scoring-mlp-german_credit_risk-v100-training.yaml
```

Ejecutar todos los experimentos:

```powershell
python src/training/train.py --config config/training/experiments/01-credit_scoring-mlp-german_credit_risk-v100-training.yaml
python src/training/train.py --config config/training/experiments/02-credit_scoring-mlp-german_credit_risk-v100-training.yaml
python src/training/train.py --config config/training/experiments/03-credit_scoring-mlp-german_credit_risk-v100-training.yaml
python src/training/train.py --config config/training/experiments/04-credit_scoring-mlp-german_credit_risk-v100-training.yaml
```

---

## 8. Artefactos generados

Los modelos y preprocesadores se guardan localmente en:

```text
python/credit_scoring/models/
```

Los reportes locales se guardan en:

```text
python/credit_scoring/reports/
```

En MLflow se registran:

```text
configs/
model/
preprocessing/
weights/
plots/
reports/
```

---

## 9. Qué revisar en MLflow

En MLflow abrir:

```text
ingenia_services/credit_scoring/training
```

O, si el experimento fue creado con el nombre usado en los YAML recientes:

```text
ingenia_services/credit_scoring/training
```

Revisar:

- `val_loss`
- `val_accuracy`
- `val_precision`
- `val_recall`
- `val_f1`
- `val_roc_auc`
- `final_val_accuracy`
- `final_val_roc_auc`

También revisar artifacts:

```text
configs/
model/
preprocessing/
reports/
plots/
weights/
```

---

## 10. Criterio para comparar modelos

Para este servicio, los criterios principales son:

```text
ROC-AUC
F1-score
Recall
Precision
Validation loss
```

Si el objetivo del negocio es reducir falsos negativos, priorizar `recall`.

Si el objetivo es reducir falsos positivos, priorizar `precision`.

---

## 11. Buenas prácticas

No modificar hiperparámetros directamente en `train.py`.

Cada variación debe vivir en un YAML nuevo dentro de:

```text
config/training/experiments/
```

Ejemplo:

```text
05-credit_scoring-mlp-german_credit_risk-v110-training.yaml
```

Cada YAML debe tener un `run_name` único.

---

## 12. Resumen

Este servicio ya sigue el estándar profesional del repositorio:

```text
YAML + MLflow central + artifacts + reportes + reproducibilidad
```

Para repetir un experimento, basta con ejecutar el mismo YAML.
