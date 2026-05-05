"""
Training Module for Xrays Evaluation - YOLO Classification

Estandarizado para:
- Ejecutarse desde python/xrays_evaluation
- Leer YAMLs desde config/training/experiments/
- Registrar experimentos en MLflow centralizado:
  ingeniia_services/.mlflow/mlflow.db
- Mantener runs locales en:
  python/xrays_evaluation/runs/
- Registrar configs, dataset summary, métricas y pesos en MLflow
"""

import os
import sys
import json
import yaml
import torch
import mlflow
import argparse
import logging as log

from pathlib import Path
from typing import Dict, Any, List, Optional

from ultralytics import YOLO, settings


# ---------------------------------------------------------
# 0. Rutas base del servicio
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

# train.py está en:
# python/xrays_evaluation/src/training/train.py
SERVICE_ROOT = CURRENT_FILE.parents[2]      # python/xrays_evaluation
PYTHON_ROOT = SERVICE_ROOT.parent           # python
REPO_ROOT = PYTHON_ROOT.parent              # ingeniia_services

if str(SERVICE_ROOT) not in sys.path:
    sys.path.append(str(SERVICE_ROOT))

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from shared.mlops.mlflow_utils import setup_mlflow_for_service


# ---------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------
def setup_logging(level=log.INFO) -> None:
    log.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[log.StreamHandler(sys.stdout)],
        force=True,
    )

    for noisy_logger in ("mlflow", "urllib3", "matplotlib"):
        log.getLogger(noisy_logger).setLevel(log.WARNING)


# ---------------------------------------------------------
# 2. Utilidades
# ---------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_service_path(raw_path: str | Path) -> Path:
    """
    Resuelve una ruta relativa desde python/xrays_evaluation.
    """
    path = Path(raw_path)

    if path.is_absolute():
        return path.resolve()

    return (SERVICE_ROOT / path).resolve()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = resolve_service_path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("El archivo YAML no contiene una configuración válida.")

    return cfg


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            if isinstance(value, (str, int, float, bool)) or value is None:
                items.append((new_key, value))
            else:
                items.append((new_key, str(value)))

    return dict(items)


def safe_log_params(params: Dict[str, Any], max_value_length: int = 250) -> None:
    """
    Registra parámetros individualmente para evitar que un parámetro duplicado
    o demasiado largo rompa todo el logging.
    """
    for key, value in params.items():
        safe_key = str(key)[:250]
        safe_value = str(value)[:max_value_length]

        try:
            mlflow.log_param(safe_key, safe_value)
        except Exception as exc:
            log.warning(f"No se pudo registrar parámetro en MLflow: {safe_key}={safe_value}. Error: {exc}")


def count_images_in_dir(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_classification_dataset_summary(dataset_path: Path) -> Dict[str, Any]:
    """
    Genera un resumen simple para datasets de clasificación estilo:
    split_data/
      train/
        class_a/
        class_b/
      val/
        class_a/
        class_b/
      test/
        class_a/
        class_b/
    """
    summary = {
        "dataset_path": str(dataset_path),
        "splits": {},
    }

    for split_name in ["train", "val", "valid", "test"]:
        split_dir = dataset_path / split_name

        if not split_dir.exists():
            continue

        class_summary = {}
        total_images = 0

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            count = count_images_in_dir(class_dir)
            class_summary[class_dir.name] = count
            total_images += count

        summary["splits"][split_name] = {
            "split_dir": str(split_dir),
            "total_images": total_images,
            "classes": class_summary,
        }

    return summary


def validate_classification_config(cfg: Dict[str, Any]) -> None:
    required_sections = [
        "project_info",
        "environment",
        "data_source",
        "model_config",
        "training_params",
        "mlops_config",
    ]

    for section in required_sections:
        if section not in cfg:
            raise KeyError(f"Falta sección requerida en YAML: {section}")

    task = str(cfg["model_config"].get("task", "")).lower().strip()

    if task != "classify":
        raise ValueError("Este script está diseñado para clasificación. Usa model_config.task: 'classify'.")


def choose_validation_split(dataset_path: Path, preferred_split: str = "test") -> str:
    """
    Para clasificación, Ultralytics puede validar contra val/test si existen.
    Si test no existe, usamos val.
    """
    preferred_dir = dataset_path / preferred_split

    if preferred_dir.exists():
        return preferred_split

    if (dataset_path / "val").exists():
        return "val"

    if (dataset_path / "valid").exists():
        return "val"

    return "val"


# ---------------------------------------------------------
# 3. Entrenamiento
# ---------------------------------------------------------
def train_yolo_classification(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    validate_classification_config(cfg)

    config_abs_path = resolve_service_path(config_path)

    mlops_runtime = setup_mlflow_for_service(
        cfg=cfg,
        current_file=__file__,
        default_service_name="xrays_evaluation",
        default_workflow_type="training",
    )

    experiment_name = mlops_runtime["experiment_name"]
    run_name = mlops_runtime["run_name"]
    standard_tags = mlops_runtime["standard_tags"]

    data_source = cfg["data_source"]
    model_config = cfg["model_config"]
    training_params = cfg["training_params"]
    environment = cfg["environment"]
    mlops_config = cfg["mlops_config"]

    dataset_path = resolve_service_path(data_source["dataset_path"])
    base_model_path = resolve_service_path(model_config["base_model_path"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {dataset_path}")

    if not base_model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo preentrenado: {base_model_path}")

    local_runs_dir = resolve_service_path(
        mlops_config.get("local_runs_dir", "runs/YOLO_CLS")
    )
    local_runs_dir.mkdir(parents=True, exist_ok=True)

    local_reports_dir = SERVICE_ROOT / "reports"
    local_reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = build_classification_dataset_summary(dataset_path)
    dataset_summary_path = local_reports_dir / f"{run_name}_dataset_summary.json"
    save_json(dataset_summary, dataset_summary_path)

    log.info("🚀 Iniciando entrenamiento YOLO classification")
    log.info(f"✔ Config path: {config_abs_path}")
    log.info(f"✔ Dataset path: {dataset_path}")
    log.info(f"✔ Base model path: {base_model_path}")
    log.info(f"✔ MLflow experiment: {experiment_name}")
    log.info(f"✔ MLflow run name: {run_name}")
    log.info(f"✔ Local runs dir: {local_runs_dir}")

    # -----------------------------------------------------
    # Configurar integración Ultralytics + MLflow
    # -----------------------------------------------------
    settings.update(
        {
            "mlflow": True,
            "runs_dir": str(local_runs_dir),
        }
    )

    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_RUN_NAME"] = run_name

    # -----------------------------------------------------
    # Cargar modelo
    # -----------------------------------------------------
    model = YOLO(str(base_model_path))

    # -----------------------------------------------------
    # Entrenamiento
    # -----------------------------------------------------
    results = model.train(
        data=str(dataset_path),
        epochs=training_params["epochs"],
        batch=training_params["batch_size"],
        imgsz=data_source["img_size"],
        patience=training_params["patience"],
        task=model_config["task"],
        device=environment["device"],
        workers=environment["workers"],
        optimizer=training_params.get("optimizer", "auto"),
        seed=environment.get("seed", 42),
        lr0=training_params.get("lr0", 0.01),
        lrf=training_params.get("lrf", 0.01),
        cos_lr=training_params.get("cos_lr", False),
        cache=training_params.get("cache", False),
        amp=training_params.get("amp", True),
        plots=training_params.get("plots", True),
        project=str(local_runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        verbose=True,
    )

    # -----------------------------------------------------
    # Validación final
    # -----------------------------------------------------
    preferred_split = cfg.get("validation_config", {}).get("split", "test")
    validation_split = choose_validation_split(dataset_path, preferred_split)

    log.info(f"📊 Ejecutando validación final en split='{validation_split}'")

    metrics = model.val(
        data=str(dataset_path),
        split=validation_split,
        imgsz=data_source["img_size"],
        batch=training_params["batch_size"],
        device=environment["device"],
        workers=environment["workers"],
        plots=True,
        project=str(local_runs_dir),
        name=f"{run_name}_final_val_{validation_split}",
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Registrar metadata adicional en MLflow
    # -----------------------------------------------------
    last_run = mlflow.last_active_run()

    if last_run:
        run_id = last_run.info.run_id
        log.info(f"🧾 Registrando metadata adicional en MLflow run_id={run_id}")

        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags(standard_tags)

            mlflow.log_artifact(str(config_abs_path), artifact_path="configs")
            mlflow.log_artifact(str(dataset_summary_path), artifact_path="dataset")

            safe_log_params(flatten_dict(cfg))

            for split_name, split_info in dataset_summary["splits"].items():
                mlflow.log_metric(
                    f"dataset_{split_name}_images_total",
                    split_info["total_images"],
                )

                for class_name, count in split_info["classes"].items():
                    safe_class_name = class_name.replace(" ", "_").replace("-", "_")
                    mlflow.log_metric(
                        f"dataset_{split_name}_{safe_class_name}_images",
                        count,
                    )

            if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
                for key, value in metrics.results_dict.items():
                    if isinstance(value, (int, float)):
                        metric_name = (
                            str(key)
                            .replace("(", "")
                            .replace(")", "")
                            .replace(" ", "_")
                            .replace("/", "_")
                        )
                        mlflow.log_metric(f"final_{metric_name}", float(value))

            save_dir = Path(getattr(results, "save_dir", local_runs_dir / run_name))
            weights_dir = save_dir / "weights"

            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"

            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")

            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path="weights")

    else:
        log.warning("⚠️ No se encontró un run activo o reciente de MLflow.")

    log.info("✅ Entrenamiento YOLO classification finalizado correctamente.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Training pipeline for Xrays Evaluation YOLO classification.")

    parser.add_argument(
        "--config",
        type=str,
        default="config/training/experiments/01-xrays_evaluation-yolo11n_cls-xrays_evaluation_img-v100-training.yaml",
        help="Path to the experiment YAML config, relative to python/xrays_evaluation.",
    )

    args = parser.parse_args()

    try:
        train_yolo_classification(args.config)

    except Exception as e:
        log.error(f"❌ Error crítico durante el entrenamiento: {e}", exc_info=True)
        sys.exit(1)


"""
Execute from:
C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services\\python\\xrays_evaluation

DVC from repo root:
cd C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services
dvc pull -r ingeniia_services_storage datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121

Training:
cd C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services\\python\\xrays_evaluation

python src/training/train.py --config config/training/experiments/01-xrays_evaluation-yolo11n_cls-xrays_evaluation_img-v100-training.yaml
python src/training/train.py --config config/training/experiments/02-xrays_evaluation-yolo11m_cls-xrays_evaluation_img-v100-training.yaml
python src/training/train.py --config config/training/experiments/03-xrays_evaluation-yolo11x_cls-xrays_evaluation_img-v100-training.yaml

MLflow UI from repo root using the same .venv:
cd C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services
python -m mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db
"""