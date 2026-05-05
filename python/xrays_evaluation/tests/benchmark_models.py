import os
import time
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

# 1. config

# 1.1 base path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))

# 1.2 models path
MODELS_DIR = os.path.join(PROJECT_ROOT, "models/YOLO")
MODELS_TO_BENCHMARK = {
    "YOLO11n-cls": os.path.join(MODELS_DIR, "xrays_evaluation_model_nano_v1.pt"),
    "YOLO11m-cls": os.path.join(MODELS_DIR, "xrays_evaluation_model_medium_v1.pt"),
    "YOLO11x-cls": os.path.join(MODELS_DIR, "xrays_evaluation_model_xlarge_v1.pt")
}

# 1.3 images path
TEST_DATA_PATH = os.path.join(PROJECT_ROOT,
                              "../../datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_test_20251130/benchmarking/")

# 1.4 MLFlow config
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "src/training/train/mlflow.db")  # Apuntamos a la misma DB de entrenamiento
MLFLOW_EXPERIMENT_NAME = "Xrays_Evaluation_Benchmarking"

# 1. 5 classes
CLASS_MAP = {0: "Anomaly", 1: "Normal"}
TARGET_NAMES = ["Anomaly", "Normal"]


# 2. utils methods
def setup_mlflow():
    """connect to MLFlow SQLite."""
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"📡 MLFlow conectado a: {tracking_uri}")


def load_test_dataset(base_path: str):
    """
    load images and tags
    """
    data = []

    # mapping
    # Anomaly = 0, Normal = 1
    folder_map = {"anomaly": 0, "normal": 1}

    for folder_name, label in folder_map.items():
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ No se encontró la carpeta: {folder_path}")

        types = ('*.jpg', '*.jpeg', '*.png')
        images = []
        for files in types:
            images.extend(glob.glob(os.path.join(folder_path, files)))

        for img_path in images:
            data.append({"path": img_path, "true_label": label})

    print(f"📂 Dataset cargado: {len(data)} imágenes encontradas.")
    return pd.DataFrame(data)


def plot_confusion_matrix(y_true, y_pred, model_name):
    """generate and save matrix confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    filename = f"cm_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename, cm


def calculate_specificity(cm):
    """calculate specificity."""
    # CM structure: [[TN, FP], [FN, TP]] si 0=Neg, 1=Pos.
    # Pero aqui 0=Anomaly, 1=Normal.
    # Si consideramos "Anomaly" como la clase positiva (enfermedad):
    # Index 0 (Anomaly): TP real
    # Index 1 (Normal): TN real
    # cm[0,0] = True Anomaly (Correctly identified sick)
    # cm[1,1] = True Normal (Correctly identified healthy)
    # cm[0,1] = False Normal (Missed the disease - Dangerous!)
    # cm[1,0] = False Anomaly (False Alarm)

    tn = cm[1, 1]
    fp = cm[1, 0]

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity


# 3. benchmarking
def evaluate_model(model_name: str, model_path: str, df_test: pd.DataFrame):
    """
    execute inference, calculate times & métrics.
    """
    print(f"\n⚡ Evaluando modelo: {model_name}...")

    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado en {model_path}, saltando...")
        return

    # load model
    model = YOLO(model_path)

    y_true = df_test['true_label'].tolist()
    y_pred = []
    inference_times = []
    total_times = []

    for _, row in df_test.iterrows():
        # execute predict
        results = model.predict(row['path'], save=False, verbose=False)

        # get class
        r = results[0]
        pred_idx = r.probs.top1
        y_pred.append(pred_idx)

        # extract time
        speed = r.speed
        inference_times.append(speed['inference'])
        total_times.append(speed['preprocess'] + speed['inference'] + speed['postprocess'])

    # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')  # Average recall

    # confusion matrix
    cm_filename, cm = plot_confusion_matrix(y_true, y_pred, model_name)
    specificity = calculate_specificity(cm)

    # time average
    avg_inference_ms = np.mean(inference_times)
    avg_total_latency_ms = np.mean(total_times)
    fps = 1000.0 / avg_total_latency_ms

    print(f"   ➤ Accuracy: {acc:.4f}")
    print(f"   ➤ Avg Inference Time: {avg_inference_ms:.2f} ms")
    print(f"   ➤ FPS Estimado: {fps:.2f}")

    # MLFlow registry
    with mlflow.start_run(run_name=f"Benchmark_{model_name}"):
        # params
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_size", len(df_test))

        # metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.log_metric("recall_average", recall)
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("avg_inference_time_ms", avg_inference_ms)
        mlflow.log_metric("avg_total_latency_ms", avg_total_latency_ms)
        mlflow.log_metric("fps", fps)

        # Efficiency Score (Custom metric: Accuracy / Latency)
        efficiency_score = acc / avg_total_latency_ms
        mlflow.log_metric("efficiency_score_index", efficiency_score)

        # Artifacts
        mlflow.log_artifact(cm_filename)

        # Confusion Matrix Raw Dictionary
        cm_dict = {
            "True_Anomaly": int(cm[0, 0]), "False_Normal": int(cm[0, 1]),
            "False_Anomaly": int(cm[1, 0]), "True_Normal": int(cm[1, 1])
        }
        mlflow.log_dict(cm_dict, "confusion_matrix.json")

    # Limpieza local
    if os.path.exists(cm_filename):
        os.remove(cm_filename)


# 4. main execution
if __name__ == '__main__':
    try:
        print("🚀 Iniciando Benchmarking de Modelos YOLO...")
        setup_mlflow()

        # load data
        df_test = load_test_dataset(TEST_DATA_PATH)

        # models
        for name, path in MODELS_TO_BENCHMARK.items():
            evaluate_model(name, path, df_test)

        print("\n✅ Benchmarking finalizado exitosamente. Revisa MLFlow UI.")

    except Exception as e:
        print(f"\n❌ Error crítico en el benchmark: {e}")

"""
usage:
# move to ingeniia_services
dvc pull -r ingeniia_services_storage datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_test_20251130
python tests/benchmark_models.py

visualize:
mlflow ui --backend-store-uri sqlite:///src/training/train/mlflow.db
"""