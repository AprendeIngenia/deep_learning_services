import os
import sys
import yaml
import torch
import mlflow
import argparse
import logging as log

from pathlib import Path
from ultralytics import YOLO, settings


log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. load config
def load_config(config_path):
    """Carga y valida el archivo YAML de configuración."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo de configuración: {path}")

    with open(path, 'r') as f:
        return yaml.safe_load(f)


# 2. train
def train_model_detection(config_path):
    # 1. Cargar Configuración
    cfg = load_config(config_path)

    # 2. paths
    CURRENT_DIR = Path(__file__).resolve().parent

    DATASET_YAML = Path(cfg['data_source']['dataset_yaml_path'])
    if not DATASET_YAML.is_absolute():
        DATASET_YAML = (CURRENT_DIR / DATASET_YAML).resolve()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"❌ No se encuentra el dataset en: {DATASET_YAML}")

    # mlflow config
    ARTIFACT_ROOT = (CURRENT_DIR / cfg['mlops_config']['artifact_root']).resolve()
    MLFLOW_DB = (CURRENT_DIR / "mlflow.db").resolve()

    # 3. config Ultralytics
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    settings.update({
        "mlflow": True,
        "runs_dir": str(ARTIFACT_ROOT)
    })

    log.info(f"🚀 Iniciando Experimento: {cfg['mlops_config']['run_name']}")
    log.info(f"📂 Configuración cargada desde: {config_path}")

    # 4. config MLFlow
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg['mlops_config']['experiment_name']
    os.environ["MLFLOW_RUN_NAME"] = cfg['mlops_config']['run_name']

    # 5. load model
    model_name = cfg['model_config']['base_model']
    model = YOLO(model_name)

    # 6. training
    results = model.train(
        data=str(DATASET_YAML),
        epochs=cfg['training_params']['epochs'],
        batch=cfg['training_params']['batch_size'],
        imgsz=cfg['data_source']['img_size'],
        patience=cfg['training_params']['patience'],
        device=cfg['environment']['device'],
        workers=cfg['environment']['workers'],
        optimizer=cfg['training_params']['optimizer'],
        seed=cfg['environment']['seed'],
        project=str(ARTIFACT_ROOT),
        name=cfg['mlops_config']['run_name'],
        exist_ok=True,
        save=True,
        verbose=True
    )

    # 7. validate
    log.info("📊 Ejecutando validación final en set de prueba...")
    metrics = model.val(split='test')

    # 8. save YAML
    last_run = mlflow.last_active_run()
    if last_run:
        with mlflow.start_run(run_id=last_run.info.run_id):
            mlflow.log_artifact(config_path, artifact_path="configs")
            mlflow.log_params(cfg['project_info'])

    log.info("✅ Entrenamiento y Registro en MLFlow completado.")
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenamiento YOLO26 con Configuración YAML")
    parser.add_argument('--config', type=str, required=True, help="Ruta al archivo .yaml de configuración")

    args = parser.parse_args()

    try:
        train_model_detection(args.config)
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        sys.exit(1)


