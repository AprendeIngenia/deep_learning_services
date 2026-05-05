"""
Training Module for Smart Checkout - YOLO Detection

Estandarizado para:
- Ejecutarse desde python/smart_checkout
- Leer YAMLs desde config/training/experiments/
- Registrar experimentos en MLflow centralizado:
  http://127.0.0.1:5000
- Mantener runs locales en:
  python/smart_checkout/runs/
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
from typing import Dict, Any, List, Optional, Union

from ultralytics import YOLO, settings


# ---------------------------------------------------------
# 0. Rutas base del servicio
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

# train.py está en:
# python/smart_checkout/src/training/train.py
SERVICE_ROOT = CURRENT_FILE.parents[2]      # python/smart_checkout
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
    Resuelve una ruta relativa desde python/smart_checkout.
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
            log.warning(
                f"No se pudo registrar parámetro en MLflow: "
                f"{safe_key}={safe_value}. Error: {exc}"
            )


def load_dataset_yaml(dataset_yaml_path: Path) -> Dict[str, Any]:
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el data.yaml: {dataset_yaml_path}")

    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if not isinstance(data_cfg, dict):
        raise ValueError("El data.yaml no contiene una configuración válida.")

    required_keys = ["train", "val", "names"]

    for key in required_keys:
        if key not in data_cfg:
            raise KeyError(f"El data.yaml debe contener la clave: {key}")

    return data_cfg


def parse_class_names(raw_names: Union[Dict[Any, Any], List[Any]]) -> Dict[int, str]:
    """
    Acepta:
    names:
      0: class_a
      1: class_b

    o:
    names: [class_a, class_b]
    """
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}

    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}

    raise ValueError("data.yaml names debe ser lista o diccionario.")


def get_dataset_root(dataset_yaml_path: Path, data_cfg: Dict[str, Any]) -> Path:
    """
    Obtiene la raíz del dataset.
    Si data.yaml tiene 'path', la usa.
    Si no, usa la carpeta donde vive data.yaml.
    """
    raw_root = data_cfg.get("path")

    if raw_root:
        root = Path(raw_root)

        if root.is_absolute():
            return root.resolve()

        return (dataset_yaml_path.parent / root).resolve()

    return dataset_yaml_path.parent.resolve()


def resolve_split_paths(
    dataset_yaml_path: Path,
    data_cfg: Dict[str, Any],
    split_name: str,
) -> List[Path]:
    if split_name not in data_cfg:
        return []

    dataset_root = get_dataset_root(dataset_yaml_path, data_cfg)
    raw_value = data_cfg[split_name]

    if isinstance(raw_value, str):
        raw_paths = [raw_value]
    elif isinstance(raw_value, list):
        raw_paths = raw_value
    else:
        raise ValueError(f"El split '{split_name}' debe ser string o lista de strings.")

    resolved = []

    for raw_path in raw_paths:
        path = Path(raw_path)

        if path.is_absolute():
            resolved.append(path.resolve())
        else:
            resolved.append((dataset_root / path).resolve())

    return resolved


def infer_labels_dir_from_images_dir(images_dir: Path) -> Path:
    """
    Inferencia común para YOLO Detect:
    images/train -> labels/train
    train/images -> train/labels
    """
    parts = list(images_dir.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)

    return images_dir.parent / "labels"


def count_images(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0

    return sum(
        1
        for file in images_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_detection_labels(labels_dir: Path, class_names: Dict[int, str]) -> Dict[str, Any]:
    """
    Cuenta anotaciones YOLO detect:
    class_id x_center y_center width height
    """
    result = {
        "labels_dir": str(labels_dir),
        "label_files": 0,
        "instances_total": 0,
        "instances_by_class": {name: 0 for name in class_names.values()},
        "empty_label_files": 0,
        "invalid_rows": 0,
    }

    if not labels_dir.exists():
        result["warning"] = "labels_dir_not_found"
        return result

    label_files = list(labels_dir.rglob("*.txt"))
    result["label_files"] = len(label_files)

    for label_file in label_files:
        content = label_file.read_text(encoding="utf-8").strip()

        if not content:
            result["empty_label_files"] += 1
            continue

        rows = content.splitlines()

        for row in rows:
            parts = row.strip().split()

            if len(parts) < 5:
                result["invalid_rows"] += 1
                continue

            try:
                class_id = int(float(parts[0]))
            except ValueError:
                result["invalid_rows"] += 1
                continue

            class_name = class_names.get(class_id, f"class_{class_id}")

            if class_name not in result["instances_by_class"]:
                result["instances_by_class"][class_name] = 0

            result["instances_by_class"][class_name] += 1
            result["instances_total"] += 1

    return result


def build_detection_dataset_summary(dataset_yaml_path: Path, data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    class_names = parse_class_names(data_cfg["names"])

    summary = {
        "dataset_yaml_path": str(dataset_yaml_path),
        "dataset_root": str(get_dataset_root(dataset_yaml_path, data_cfg)),
        "classes": class_names,
        "splits": {},
    }

    for split_name in ["train", "val", "valid", "test"]:
        image_dirs = resolve_split_paths(dataset_yaml_path, data_cfg, split_name)

        if not image_dirs:
            continue

        split_summary = {
            "image_dirs": [str(p) for p in image_dirs],
            "images_total": 0,
            "labels": [],
        }

        for images_dir in image_dirs:
            labels_dir = infer_labels_dir_from_images_dir(images_dir)

            image_count = count_images(images_dir)
            label_summary = count_detection_labels(labels_dir, class_names)

            split_summary["images_total"] += image_count
            split_summary["labels"].append(
                {
                    "images_dir": str(images_dir),
                    "images_count": image_count,
                    **label_summary,
                }
            )

        summary["splits"][split_name] = split_summary

    return summary


def validate_detection_config(cfg: Dict[str, Any]) -> None:
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

    if task != "detect":
        raise ValueError("Este script está diseñado para detección. Usa model_config.task: 'detect'.")


def choose_validation_split(data_cfg: Dict[str, Any], preferred_split: str = "test") -> str:
    if preferred_split in data_cfg:
        return preferred_split

    if "val" in data_cfg:
        return "val"

    if "valid" in data_cfg:
        return "val"

    return "val"


# ---------------------------------------------------------
# 3. Entrenamiento
# ---------------------------------------------------------
def train_model_detection(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    validate_detection_config(cfg)

    config_abs_path = resolve_service_path(config_path)

    mlops_runtime = setup_mlflow_for_service(
        cfg=cfg,
        current_file=__file__,
        default_service_name="smart_checkout",
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

    dataset_yaml_path = resolve_service_path(data_source["dataset_yaml_path"])

    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset YAML: {dataset_yaml_path}")

    data_cfg = load_dataset_yaml(dataset_yaml_path)
    dataset_summary = build_detection_dataset_summary(dataset_yaml_path, data_cfg)

    local_runs_dir = resolve_service_path(
        mlops_config.get("local_runs_dir", "runs/YOLO_DETECT")
    )
    local_runs_dir.mkdir(parents=True, exist_ok=True)

    local_reports_dir = SERVICE_ROOT / "reports"
    local_reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary_path = local_reports_dir / f"{run_name}_dataset_summary.json"
    save_json(dataset_summary, dataset_summary_path)

    log.info("🚀 Iniciando entrenamiento YOLO detection")
    log.info(f"✔ Config path: {config_abs_path}")
    log.info(f"✔ Dataset YAML: {dataset_yaml_path}")
    log.info(f"✔ MLflow experiment: {experiment_name}")
    log.info(f"✔ MLflow run name: {run_name}")
    log.info(f"✔ Local runs dir: {local_runs_dir}")

    settings.update(
        {
            "mlflow": True,
            "runs_dir": str(local_runs_dir),
        }
    )

    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_RUN_NAME"] = run_name

    model_name = model_config["base_model"]
    model = YOLO(model_name)

    results = model.train(
        data=str(dataset_yaml_path),
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
        close_mosaic=training_params.get("close_mosaic", 10),
        cache=training_params.get("cache", False),
        amp=training_params.get("amp", True),
        plots=training_params.get("plots", True),
        project=str(local_runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        verbose=True,
    )

    preferred_split = cfg.get("validation_config", {}).get("split", "test")
    validation_split = choose_validation_split(data_cfg, preferred_split)

    log.info(f"📊 Ejecutando validación final en split='{validation_split}'")

    metrics = model.val(
        data=str(dataset_yaml_path),
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
                    split_info["images_total"],
                )

                instances_total = 0

                for label_info in split_info["labels"]:
                    instances_total += label_info.get("instances_total", 0)

                    for class_name, count in label_info.get("instances_by_class", {}).items():
                        safe_class_name = class_name.replace(" ", "_").replace("-", "_")
                        mlflow.log_metric(
                            f"dataset_{split_name}_{safe_class_name}_instances",
                            count,
                        )

                mlflow.log_metric(
                    f"dataset_{split_name}_instances_total",
                    instances_total,
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

    log.info("✅ Entrenamiento YOLO detection finalizado correctamente.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Training pipeline for Smart Checkout YOLO detection."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training/experiments/01-smart_checkout-yolo26s_detect-smart_checkout_img-v100-training.yaml",
        help="Path to the experiment YAML config, relative to python/smart_checkout.",
    )

    args = parser.parse_args()

    try:
        train_model_detection(args.config)

    except Exception as e:
        log.error(f"❌ Error crítico durante el entrenamiento: {e}", exc_info=True)
        sys.exit(1)


"""
Execute from:
C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services\\python\\smart_checkout

DVC from repo root:
cd C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services
dvc pull -r ingeniia_services_storage datasets/ingeniia_services_smart_checkout_img_v1.0.0_training_20260120

Training:
cd C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services\\python\\smart_checkout

python src/training/train.py --config config/training/experiments/01-smart_checkout-yolo26s_detect-smart_checkout_img-v100-training.yaml

MLflow server from repo root:
cd C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services
python -m mlflow server `
  --backend-store-uri sqlite:///.mlflow/mlflow.db `
  --default-artifact-root file:///C:/Users/santi/Desktop/proyectos/ingeniia_services/.mlflow/artifacts `
  --host 127.0.0.1 `
  --port 5000
"""