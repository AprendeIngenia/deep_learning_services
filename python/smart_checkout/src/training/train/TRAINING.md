# 🛒 Smart Checkout - Módulo de Entrenamiento (YOLO26)

Este módulo gestiona el ciclo de vida del entrenamiento para el sistema de detección de productos del Cajero Automático Inteligente (**Smart Checkout**). Utiliza **Ultralytics YOLO26** como arquitectura base y **MLFlow** para el registro de experimentos, métricas y trazabilidad (MLOps).

---

## 📋 Tabla de Contenidos
1. [Requisitos Previos](#-requisitos-previos)
2. [Estructura del Proyecto](#-estructura-del-proyecto)
3. [Configuración de Experimentos (YAML)](#-configuración-de-experimentos-yaml)
4. [Ejecución del Entrenamiento](#-ejecución-del-entrenamiento)
5. [Seguimiento con MLFlow](#-seguimiento-con-mlflow)
6. [Resultados y Modelos](#-resultados-y-modelos)

---

## 🛠 Requisitos Previos

Asegúrate de tener un entorno virtual activo (Python 3.10+) y las dependencias instaladas.

Desde la raíz del proyecto:
```bash
# Instalación de dependencias en CPU
pip install -r requirements-cpu.txt

# Instalación de dependencias en GPU
pip install -r requirements-gpu.txt
```
Hardware Recomendado:

- GPU: NVIDIA con soporte CUDA (VRAM >= 6GB recomendada para YOLO26s).
- RAM: 16GB+.

---

## 📂 Estructura del Entrenamiento
```text
src/training/train/
├── experiments/          # ⚙️ Archivos YAML de configuración de experimentos
│   └── 01_yolo26s.yaml   # Ejemplo de experimento base
├── runs/                 # 📂 Salida de Ultralytics (pesos, gráficas locales)
├── main.py               # 🚀 Script principal de ejecución (Entry Point)
├── mlflow.db             # 🗄️ Base de datos SQLite local de MLFlow
└── TRAINING.md           # 📄 Este archivo
```
---

## ⚙️ Configuración de Experimentos (YAML)
Para garantizar la reproducibilidad, cada entrenamiento se define en un archivo .yaml dentro de la carpeta experiments/. No debes modificar el código (main.py) para cambiar hiperparámetros.

Ejemplo de estructura `(experiments/tu_experimento.yaml)`:
```YAML
version: 1.0.0

project_info:
  name: "SmartCheckout_Detection"
  experiment_id: "EXP-001"
  tags: ["detection", "yolo26", "produccion"]

environment:
  device: 0       # ID de la GPU (0, 1) o 'cpu'
  seed: 42        # Semilla para resultados deterministas
  workers: 8      # Hilos para carga de datos

data_source:
  # Ruta al data.yaml generado por Roboflow
  dataset_yaml_path: "../datasets/smart_checkout/data.yaml"
  img_size: 640

model_config:
  base_model: "yolo26s.pt" # Puede ser n, s, m, l, x
  task: "detect"

training_params:
  epochs: 100
  batch_size: 64
  patience: 15    # Early Stopping
  optimizer: "auto"
```
---

## 🚀 Ejecución del Entrenamiento
Para iniciar un entrenamiento, ejecuta el script main.py pasando el argumento --config con la ruta de tu archivo YAML.

Desde la raíz del proyecto:
```bash
# Ejemplo de ejecución
python src/training/train/main.py --config src/training/train/experiments/01_yolo26s-training.yaml
```

Lo que verás en la consola:

1. Carga de configuración y validación de rutas.
2. Inicialización de MLFlow.
3. Descarga de pesos pre-entrenados (si es la primera vez).
4. Barra de progreso del entrenamiento (Epochs).
---

## 📊 Seguimiento con MLFlow
Este proyecto utiliza MLFlow para registrar métricas en tiempo real (Pérdida, mAP, Precision, Recall).

Para visualizar el dashboard de experimentos:

1. Abre una terminal en la carpeta src/training/train/.

2. Ejecuta el servidor de interfaz gráfica:
   ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
   
3. Abre tu navegador en: `http://127.0.0.1:5000`

¿Qué encontrarás en MLFlow?

- **Parámetros:** Todos los valores del YAML usados.
- **Métricas:** Gráficas de train/box_loss, metrics/mAP50-95, etc.
- **Artefactos:** El archivo .yaml original usado (para auditoría) y el modelo final (si se configura el log de modelos).

---

## 🏆 Resultados y Modelos

Al finalizar el entrenamiento, los artefactos se generan en dos ubicaciones:
1. **Carpeta** `runs/YOLO26/{nombre_run}/`:
    - `weights/best.pt`: El mejor modelo obtenido (¡Este es el que usarás en producción!).
    - `weights/last.pt`: El último checkpoint.
    - `confusion_matrix.png`: Matriz de confusión.
    - `results.csv`: Histórico de métricas.

2. **MLFlow Artifacts:**
    - Copia de seguridad de la configuración.
    - Logs detallados.
