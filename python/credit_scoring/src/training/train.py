"""
Training Module for Credit Scoring MLP

Estandarizado para:
- Ejecutarse desde python/credit_scoring
- Leer YAMLs desde config/training/experiments/
- Registrar experimentos en MLflow centralizado:
  ingeniia_services/.mlflow/mlflow.db
- Mantener modelos y reportes locales en:
  python/credit_scoring/models/
  python/credit_scoring/reports/
"""

import os
import sys
import yaml
import math
import torch
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
import mlflow.pytorch
import torch.nn as nn
import logging as log
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from mlflow.models.signature import infer_signature


# ---------------------------------------------------------
# 0. Rutas base del servicio
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

# train.py está en:
# python/credit_scoring/src/training/train.py
SERVICE_ROOT = CURRENT_FILE.parents[2]      # python/credit_scoring
PYTHON_ROOT = SERVICE_ROOT.parent           # python
REPO_ROOT = PYTHON_ROOT.parent              # ingeniia_services

# Importar módulos internos del servicio credit_scoring
if str(SERVICE_ROOT) not in sys.path:
    sys.path.append(str(SERVICE_ROOT))

# Importar módulos compartidos del repositorio
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from src.processing.main import CreditDataPreprocessor
from src.training.model import CreditScoringModel
from shared.mlops.mlflow_utils import setup_mlflow_for_service


# ---------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------
def setup_logging(level=log.INFO, log_file: Optional[str] = None) -> None:
    handlers = [log.StreamHandler(sys.stdout)]

    if log_file:
        from logging.handlers import RotatingFileHandler

        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
            )
        )

    log.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    for noisy_logger in ("mlflow", "urllib3", "matplotlib"):
        log.getLogger(noisy_logger).setLevel(log.WARNING)


# ---------------------------------------------------------
# 2. Utilidades
# ---------------------------------------------------------
def resolve_service_path(raw_path: str | Path) -> Path:
    """
    Resuelve una ruta relativa desde python/credit_scoring.

    Ejemplo:
    ../../datasets/...
    se interpreta desde:
    python/credit_scoring
    """
    path = Path(raw_path)

    if path.is_absolute():
        return path.resolve()

    return (SERVICE_ROOT / path).resolve()


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Aplana un diccionario anidado para registrar parte de la configuración en MLflow.
    """
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
    Registra parámetros en MLflow evitando valores demasiado largos.
    """
    safe_params = {}

    for key, value in params.items():
        safe_key = str(key)[:250]
        safe_value = str(value)[:max_value_length]
        safe_params[safe_key] = safe_value

    if safe_params:
        mlflow.log_params(safe_params)


# ---------------------------------------------------------
# 3. Clase principal de entrenamiento
# ---------------------------------------------------------
class CreditScoringModelTraining:
    def __init__(self, config_path: Path) -> None:
        self.config_path = resolve_service_path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de configuración: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.params = yaml.safe_load(f)

        if not isinstance(self.params, dict):
            raise ValueError("El archivo YAML no contiene una configuración válida.")

        log.info("--- Config Training ---")
        log.info(f"✔ Config path: {self.config_path}")
        log.info(f"✔ Service root: {SERVICE_ROOT}")
        log.info(f"✔ Repo root: {REPO_ROOT}")

        # -------------------------------------------------
        # MLflow runtime centralizado
        # -------------------------------------------------
        self.mlops_runtime = setup_mlflow_for_service(
            cfg=self.params,
            current_file=__file__,
            default_service_name="credit_scoring",
            default_workflow_type="training",
        )

        self.mlflow_project_name = self.mlops_runtime["experiment_name"]
        self.mlflow_run_name = self.mlops_runtime["run_name"]
        self.mlflow_standard_tags = self.mlops_runtime["standard_tags"]

        # -------------------------------------------------
        # Paths
        # -------------------------------------------------
        data_path_cfg = self.params["data_source"]["data_path"]

        self.dataset_path = resolve_service_path(data_path_cfg["dataset_path"])
        self.artifact_name_or_path = data_path_cfg["artifact_path"]
        self.preprocessor_filename = data_path_cfg["preprocessor_filename"]

        self.local_models_dir = SERVICE_ROOT / "models"
        self.local_models_dir.mkdir(parents=True, exist_ok=True)

        self.local_artifacts_dir = SERVICE_ROOT / "reports"
        self.local_artifacts_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"✔ Dataset path: {self.dataset_path}")
        log.info(f"✔ Local models dir: {self.local_models_dir}")
        log.info(f"✔ Local reports dir: {self.local_artifacts_dir}")

        # -------------------------------------------------
        # Arquitectura
        # -------------------------------------------------
        model_cfg = self.params["model_config"]["architecture"]

        self.hidden_layers = model_cfg["hidden_layers"]
        self.use_batch_norm = model_cfg["use_batch_norm"]
        self.activation_fn = model_cfg["activation_fn"]
        self.dropout_rate = model_cfg["dropout_rate"]

        self.model_name = self.params["model_config"]["model_name"]

        # -------------------------------------------------
        # Training config
        # -------------------------------------------------
        train_cfg = self.params["training_params"]

        self.optimizer_name = train_cfg["optimizer"]["name"]
        self.learning_rate = train_cfg["optimizer"]["learning_rate"]
        self.weight_decay = train_cfg["optimizer"].get("weight_decay", 0.0)

        self.use_pos_weight = train_cfg["loss_function"]["use_pos_weight"]

        self.scheduler_patience = train_cfg["scheduler"]["patience"]
        self.scheduler_factor = train_cfg["scheduler"]["factor"]

        self.epochs = train_cfg["epochs"]
        self.batch_size = train_cfg["batch_size"]

        self.test_size = train_cfg["test_size"]
        self.random_state = train_cfg["random_state"]

        self.early_stopping_patience = train_cfg["early_stopping"]["patience"]
        self.early_stopping_delta = train_cfg["early_stopping"]["delta"]

        # -------------------------------------------------
        # Reproducibilidad
        # -------------------------------------------------
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        # -------------------------------------------------
        # Device
        # -------------------------------------------------
        self.device = self._resolve_device()

        # -------------------------------------------------
        # Instancias auxiliares
        # -------------------------------------------------
        self.data_preprocessor = CreditDataPreprocessor()

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_auc": [],
            "val_auc": [],
        }

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    def _resolve_device(self) -> torch.device:
        env_device = str(self.params.get("environment", {}).get("device", "auto")).lower()

        if env_device == "cpu":
            return torch.device("cpu")

        if env_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if env_device.isdigit() and torch.cuda.is_available():
            return torch.device(f"cuda:{env_device}")

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga el dataset y lo divide en entrenamiento y validación.
        """
        log.info("--- Load data ---")
        log.info(f"✔ Loading data from: {self.dataset_path}")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"No se encontró el dataset: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
            log.info("✔ Columna 'Unnamed: 0' eliminada del DataFrame.")

        log.info("✔ Splitting data into training and validation sets.")

        df_train, df_val = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.data_preprocessor.target_feature],
        )

        return df_train, df_val

    def _preprocess_data(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ajusta el preprocesador con train y transforma train/val.
        """
        log.info("--- Preprocessing data ---")

        preprocessor = self.data_preprocessor.fit_preprocessor(df_train)

        x_train_processed, y_train = self.data_preprocessor.process_data(df_train, preprocessor)
        x_val_processed, y_val = self.data_preprocessor.process_data(df_val, preprocessor)

        x_train_tensor = torch.tensor(x_train_processed, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        x_val_tensor = torch.tensor(x_val_processed, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        path_preprocessor = self.local_models_dir / self.preprocessor_filename
        joblib.dump(preprocessor, path_preprocessor)

        log.info(f"✔ Preprocessor saved to: {path_preprocessor}")

        return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor

    # -----------------------------------------------------
    # Métricas
    # -----------------------------------------------------
    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calcula accuracy, precision, recall, f1 y ROC-AUC.
        """
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        }

    def _evaluate_split(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Evalúa un split completo.
        """
        model.eval()

        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y).item()

            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            y_true = y.detach().cpu().numpy().reshape(-1)

            metrics = self._compute_metrics(y_true, prob, threshold=0.5)
            metrics["loss"] = loss

        return metrics

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    def _plot_and_save(
        self,
        xs: List[int],
        ys1: List[float],
        ys2: List[float],
        title: str,
        ylabel: str,
        filename: str,
    ) -> Path:
        plt.figure()
        plt.plot(xs, ys1, label="train")
        plt.plot(xs, ys2, label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

        out = self.local_artifacts_dir / filename

        plt.savefig(out, bbox_inches="tight")
        plt.close()

        return out

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str,
    ) -> Path:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (val)")
        plt.colorbar()

        tick_marks = [0, 1]
        plt.xticks(tick_marks, ["bad(0)", "good(1)"])
        plt.yticks(tick_marks, ["bad(0)", "good(1)"])

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        out = self.local_artifacts_dir / filename

        plt.savefig(out, bbox_inches="tight")
        plt.close()

        return out

    def _plot_roc_pr(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        roc_file: str,
        pr_file: str,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        roc_path = None
        pr_path = None

        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)

            plt.figure()
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (val)")

            roc_path = self.local_artifacts_dir / roc_file

            plt.savefig(roc_path, bbox_inches="tight")
            plt.close()

        except ValueError:
            log.warning("No se pudo generar ROC curve.")

        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)

            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve (val)")

            pr_path = self.local_artifacts_dir / pr_file

            plt.savefig(pr_path, bbox_inches="tight")
            plt.close()

        except ValueError:
            log.warning("No se pudo generar Precision-Recall curve.")

        return roc_path, pr_path

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------
    def _setup_loss_function(self, y_train: torch.Tensor) -> nn.Module:
        """
        Configura la función de pérdida.
        """
        if self.use_pos_weight:
            y_train_cpu = y_train.detach().cpu().numpy().reshape(-1)

            pos = float(np.sum(y_train_cpu == 1))
            neg = float(np.sum(y_train_cpu == 0))

            if pos == 0 or neg == 0:
                log.warning(
                    "Una de las clases no está presente en entrenamiento. "
                    "No se usará pos_weight."
                )
                return nn.BCEWithLogitsLoss()

            pos_weight_value = neg / pos
            pos_weight_tensor = torch.tensor(
                [pos_weight_value],
                dtype=torch.float32,
                device=self.device,
            )

            log.info(f"✔ Using weighted BCE loss with pos_weight={pos_weight_value:.4f}")

            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        log.info("✔ Using standard BCE loss.")
        return nn.BCEWithLogitsLoss()

    # -----------------------------------------------------
    # Entrenamiento
    # -----------------------------------------------------
    def _run_training_loop(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> int:
        """
        Ejecuta el loop de entrenamiento con early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        epochs_run = 0

        best_model_path = self.local_models_dir / self.model_name

        log.info("--- Starting training loop ---")

        for epoch in range(self.epochs):
            model.train()
            epochs_run = epoch + 1

            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i : i + self.batch_size]
                y_batch = y_train[i : i + self.batch_size]

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            train_metrics = self._evaluate_split(model, x_train, y_train, criterion)
            val_metrics = self._evaluate_split(model, x_val, y_val, criterion)

            if scheduler is not None:
                scheduler.step(val_metrics["loss"])

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_auc"].append(train_metrics["roc_auc"])
            self.history["val_auc"].append(val_metrics["roc_auc"])

            current_lr = optimizer.param_groups[0]["lr"]

            log.info(
                f"✔ Epoch [{epoch + 1}/{self.epochs}] "
                f"| TrainLoss: {train_metrics['loss']:.4f} "
                f"| ValLoss: {val_metrics['loss']:.4f} "
                f"| TrainAcc: {train_metrics['accuracy']:.4f} "
                f"| ValAcc: {val_metrics['accuracy']:.4f} "
                f"| TrainAUC: {train_metrics['roc_auc']:.4f} "
                f"| ValAUC: {val_metrics['roc_auc']:.4f} "
                f"| LR: {current_lr:.6f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "train_precision": train_metrics["precision"],
                    "val_precision": val_metrics["precision"],
                    "train_recall": train_metrics["recall"],
                    "val_recall": val_metrics["recall"],
                    "train_f1": train_metrics["f1"],
                    "val_f1": val_metrics["f1"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "lr": current_lr,
                },
                step=epoch,
            )

            if val_metrics["loss"] < best_val_loss - self.early_stopping_delta:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0

                torch.save(model.state_dict(), best_model_path)

                log.info(f"✔ Best model updated: {best_model_path}")

            else:
                patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    log.info(f"✘ Early stopping activado en epoch {epoch + 1}.")
                    break

        log.info("--- Training finished ---")

        return epochs_run

    # -----------------------------------------------------
    # MLflow helpers
    # -----------------------------------------------------
    def _log_config_params(self) -> None:
        """
        Registra configuración YAML aplanada.
        """
        flat_params = flatten_dict(self.params)
        safe_log_params(flat_params)

    def _log_basic_params(self, num_features: int) -> None:
        """
        Registra parámetros principales del entrenamiento.
        """
        mlflow.log_params(
            {
                "num_features": num_features,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "hidden_layers": str(self.hidden_layers),
                "use_batch_norm": self.use_batch_norm,
                "activation_fn": self.activation_fn,
                "dropout_rate": self.dropout_rate,
                "optimizer": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "use_pos_weight": self.use_pos_weight,
                "scheduler_patience": self.scheduler_patience,
                "scheduler_factor": self.scheduler_factor,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_delta": self.early_stopping_delta,
            }
        )

        tags = self.params.get("project_info", {}).get("tags", [])

        if isinstance(tags, list):
            for i, tag in enumerate(tags):
                mlflow.set_tag(f"project.tag.{i}", str(tag))

    def _generate_and_log_performance_report(
        self,
        model: CreditScoringModel,
        final_metrics: Dict[str, float],
        num_features: int,
        epochs_run: int,
        run_name: str,
    ) -> None:
        """
        Genera un reporte YAML local y lo registra en MLflow.
        """
        log.info("--- Generating performance report ---")

        model_info = model.get_model_info()

        report_data = {
            "experiment_id": self.params.get("project_info", {}).get("experiment_id", "N/A"),
            "service_name": self.params.get("project_info", {}).get("service_name", "credit_scoring"),
            "workflow_type": self.params.get("project_info", {}).get("workflow_type", "training"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "dataset_name": self.params.get("data_source", {}).get("dataset_name", "N/A"),
                "dataset_version": self.params.get("data_source", {}).get("dataset_version", "N/A"),
                "dataset_path": str(self.dataset_path),
            },
            "model_architecture": {
                "model_type": model_info["model_type"],
                "input_features": num_features,
                "hidden_layers": model_info["architecture"]["hidden_layers"],
                "use_batch_norm": model_info["use_batch_norm"],
                "activation_fn": model_info["activation_fn"],
                "dropout_rate": model_info["dropout_rate"],
                "output_layer_neurons": model_info["architecture"]["output_layer"],
                "total_parameters": model_info["total_parameters"],
            },
            "training_configuration": {
                "optimizer": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "loss_function": "BCEWithLogitsLoss",
                "use_pos_weight": self.use_pos_weight,
                "epochs_run": epochs_run,
                "batch_size": self.batch_size,
            },
            "final_validation_metrics": {
                key: round(value, 4)
                for key, value in final_metrics.items()
                if not math.isnan(value)
            },
        }

        report_filename = f"{run_name}_performance_report.yaml"
        report_path = self.local_artifacts_dir / report_filename

        with open(report_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report_data, f, indent=2, sort_keys=False, allow_unicode=True)

        log.info(f"✔ Performance report saved locally to: {report_path}")

        mlflow.log_artifact(str(report_path), artifact_path="reports")

        log.info("✔ Performance report logged to MLflow artifacts.")

    def _log_model_with_signature(self, model: nn.Module, x_example: torch.Tensor) -> None:
        """
        Registra el modelo PyTorch en MLflow con signature e input example.
        """
        model_cpu = model.to("cpu").eval()

        with torch.no_grad():
            y_example = model_cpu(x_example).numpy()

        signature = infer_signature(x_example.numpy(), y_example)
        input_example = x_example.numpy()

        torch_version = torch.__version__.split("+")[0]

        pip_requirements = [
            f"torch=={torch_version}",
            "scikit-learn",
            "pandas",
            "numpy",
        ]

        try:
            mlflow.pytorch.log_model(
                model_cpu,
                name=self.artifact_name_or_path,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
            )

        except TypeError:
            mlflow.pytorch.log_model(
                model_cpu,
                artifact_path=self.artifact_name_or_path,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
            )

    def _log_plots_and_reports(
        self,
        y_true_val: np.ndarray,
        y_prob_val: np.ndarray,
    ) -> None:
        """
        Genera plots y reportes de validación y los registra en MLflow.
        """
        epochs = list(range(1, len(self.history["train_loss"]) + 1))

        loss_png = self._plot_and_save(
            epochs,
            self.history["train_loss"],
            self.history["val_loss"],
            "Training vs Validation Loss",
            "Loss",
            "loss_train_val.png",
        )

        acc_png = self._plot_and_save(
            epochs,
            self.history["train_acc"],
            self.history["val_acc"],
            "Training vs Validation Accuracy",
            "Accuracy",
            "acc_train_val.png",
        )

        roc_png, pr_png = self._plot_roc_pr(
            y_true_val,
            y_prob_val,
            roc_file="roc_val.png",
            pr_file="pr_val.png",
        )

        y_pred_val = (y_prob_val >= 0.5).astype(int)

        cm_png = self._plot_confusion_matrix(
            y_true_val,
            y_pred_val,
            filename="confusion_matrix_val.png",
        )

        cls_report = classification_report(
            y_true_val,
            y_pred_val,
            target_names=["bad(0)", "good(1)"],
            zero_division=0,
        )

        report_path = self.local_artifacts_dir / "classification_report_val.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(cls_report)

        mlflow.log_artifact(str(loss_png), artifact_path="plots")
        mlflow.log_artifact(str(acc_png), artifact_path="plots")
        mlflow.log_artifact(str(cm_png), artifact_path="plots")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        if roc_png:
            mlflow.log_artifact(str(roc_png), artifact_path="plots")

        if pr_png:
            mlflow.log_artifact(str(pr_png), artifact_path="plots")

    # -----------------------------------------------------
    # Orquestador principal
    # -----------------------------------------------------
    def train(self) -> None:
        """
        Ejecuta el pipeline completo de entrenamiento.
        """
        log.info(f"✔ Hardware used: {self.device}")
        log.info(f"✔ MLflow experiment: {self.mlflow_project_name}")
        log.info(f"✔ MLflow run name: {self.mlflow_run_name}")

        mlflow.set_experiment(self.mlflow_project_name)

        with mlflow.start_run(run_name=self.mlflow_run_name):
            mlflow.set_tags(self.mlflow_standard_tags)

            mlflow.log_artifact(str(self.config_path), artifact_path="configs")

            self._log_config_params()

            log.info("--- Init Training ---")

            # 1. Load and split data
            df_train, df_val = self._load_and_split_data()

            # 2. Preprocess data
            x_train, y_train, x_val, y_val = self._preprocess_data(df_train, df_val)

            num_features = x_train.shape[1]

            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

            # 3. Configure model
            log.info(f"✔ Initializing model with hidden layers: {self.hidden_layers}")
            log.info(f"✔ Number of input features: {num_features}")

            model = CreditScoringModel(
                num_features=num_features,
                hidden_layers=self.hidden_layers,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
                activation_fn=self.activation_fn,
            ).to(self.device)

            # 4. Loss
            criterion = self._setup_loss_function(y_train)

            # 5. Optimizer
            optimizer_name = self.optimizer_name.lower()

            if optimizer_name == "adam":
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

            elif optimizer_name == "adamw":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

            elif optimizer_name == "sgd":
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

            else:
                raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

            log.info(f"✔ Using optimizer: {self.optimizer_name} with lr={self.learning_rate}")

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )

            # 6. MLflow params
            self._log_basic_params(num_features=num_features)

            # 7. Training loop
            epochs_run = self._run_training_loop(
                model,
                criterion,
                optimizer,
                scheduler,
                x_train,
                y_train,
                x_val,
                y_val,
            )

            # 8. Load best model
            best_model_path = self.local_models_dir / self.model_name

            if not best_model_path.exists():
                raise FileNotFoundError(f"No se encontró el modelo entrenado: {best_model_path}")

            log.info(f"✔ Loading best model from: {best_model_path}")

            model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            model.eval()

            # 9. Final validation metrics
            with torch.no_grad():
                logits_val = model(x_val)
                prob_val = torch.sigmoid(logits_val).detach().cpu().numpy().reshape(-1)
                y_val_np = y_val.detach().cpu().numpy().reshape(-1)

            final_metrics = self._compute_metrics(y_val_np, prob_val, threshold=0.5)

            mlflow.log_metrics(
                {
                    f"final_val_{key}": value
                    for key, value in final_metrics.items()
                    if not math.isnan(value)
                }
            )

            # 10. Plots and reports
            self._log_plots_and_reports(y_val_np, prob_val)

            self._generate_and_log_performance_report(
                model=model,
                final_metrics=final_metrics,
                num_features=num_features,
                epochs_run=epochs_run,
                run_name=self.mlflow_run_name,
            )

            # 11. Log model
            x_example = x_train[:5].detach().cpu()

            self._log_model_with_signature(model, x_example)

            # 12. Log local artifacts required by inference
            preprocessor_path = self.local_models_dir / self.preprocessor_filename

            if preprocessor_path.exists():
                mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing")

            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), artifact_path="weights")

            log.info("✔ Preprocessor and model saved in MLflow.")
            log.info("✅ Training completed successfully.")


# ---------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Training pipeline for Credit Scoring MLP."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training/experiments/01-credit_scoring-mlp-german_credit_risk-v100-training.yaml",
        help="Path to the experiment YAML config, relative to python/credit_scoring.",
    )

    cli_args = parser.parse_args()

    log.info(f"Config path: {cli_args.config}")

    try:
        trainer = CreditScoringModelTraining(Path(cli_args.config))
        trainer.train()

    except Exception as e:
        log.error(f"Error running the training: {e}", exc_info=True)
        sys.exit(1)


"""
Execute from:
C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services\\python\\credit_scoring

Commands:
from repo root:
ingeniia_services:
dvc pull -r ingeniia_services_storage datasets/ingeniia_services_credit_scoring_csv_german_credit_risk_v1.0.0_training_20250825

python src/training/train.py --config config/training/experiments/01-credit_scoring-mlp-german_credit_risk-v100-training.yaml
python src/training/train.py --config config/training/experiments/02-credit_scoring-mlp-german_credit_risk-v100-training.yaml
python src/training/train.py --config config/training/experiments/03-credit_scoring-mlp-german_credit_risk-v100-training.yaml
python src/training/train.py --config config/training/experiments/04-credit_scoring-mlp-german_credit_risk-v100-training.yaml

MLflow UI from repo root:
cd C:\\Users\\santi\\Desktop\\proyectos\\ingeniia_services
mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db
"""