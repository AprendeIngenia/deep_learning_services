import os
import mlflow
import logging as log

from pathlib import Path
from typing import Dict, Any, Optional
from mlflow.tracking import MlflowClient


def find_repo_root(start_path: Path) -> Path:
    """
    Busca la raíz del repositorio ingeniia_services.
    Criterio: debe contener las carpetas python/ y datasets/.
    """
    start_path = start_path.resolve()

    for parent in [start_path, *start_path.parents]:
        if (parent / "python").exists() and (parent / "datasets").exists():
            return parent

    raise RuntimeError(
        "No se pudo encontrar la raíz del repositorio. "
        "Asegúrate de ejecutar el script dentro de ingeniia_services."
    )


def get_tracking_uri(repo_root: Path, tracking_uri: Optional[str] = None) -> str:
    """
    Devuelve el tracking URI central.

    Prioridad:
    1. Valor explícito en YAML si no es 'auto'
    2. Variable de entorno MLFLOW_TRACKING_URI
    3. SQLite local centralizado como fallback de desarrollo
    """
    if tracking_uri and tracking_uri != "auto":
        return tracking_uri

    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if env_tracking_uri:
        return env_tracking_uri

    mlflow_dir = repo_root / ".mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    db_path = mlflow_dir / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def get_artifact_location(repo_root: Path, service_name: str, workflow_type: str) -> str:
    """
    Define el artifact store local centralizado para cada servicio/workflow.
    """
    artifact_dir = repo_root / ".mlflow" / "artifacts" / service_name / workflow_type
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.resolve().as_uri()


def ensure_experiment(experiment_name: str, artifact_location: str) -> str:
    """
    Crea el experimento si no existe y devuelve su experiment_id.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location
        )
        log.info(f"✅ Experimento creado: {experiment_name}")
        log.info(f"📦 Artifact location: {artifact_location}")
        return experiment_id

    log.info(f"✅ Experimento encontrado: {experiment_name}")
    return experiment.experiment_id


def build_experiment_name(repo_name: str, service_name: str, workflow_type: str) -> str:
    return f"{repo_name}/{service_name}/{workflow_type}"


def build_standard_tags(cfg: Dict[str, Any], service_name: str, workflow_type: str) -> Dict[str, str]:
    project_info = cfg.get("project_info", {})
    model_config = cfg.get("model_config", {})
    data_source = cfg.get("data_source", {})

    tags = {
        "repo.name": "ingeniia_services",
        "service.name": service_name,
        "workflow.type": workflow_type,
        "task.type": str(model_config.get("task", "unknown")),
        "model.family": str(model_config.get("model_family", model_config.get("base_model", "unknown"))),
        "model.name": str(model_config.get("model_name", model_config.get("base_model", "unknown"))),
        "dataset.name": str(data_source.get("dataset_name", "unknown")),
        "dataset.version": str(data_source.get("dataset_version", "unknown")),
        "experiment.id": str(project_info.get("experiment_id", "unknown")),
        "project.stage": str(project_info.get("stage", "unknown")),
    }

    extra_tags = project_info.get("tags", [])

    if isinstance(extra_tags, list):
        for idx, tag in enumerate(extra_tags):
            tags[f"project.tag.{idx}"] = str(tag)

    return tags


def setup_mlflow_for_service(cfg: Dict[str, Any], current_file: str, default_service_name: str,
                             default_workflow_type: str = "training") -> Dict[str, Any]:
    """
    Configura MLflow de forma homogénea para cualquier servicio.
    """
    current_path = Path(current_file).resolve()
    repo_root = find_repo_root(current_path)

    project_info = cfg.get("project_info", {})
    mlops_config = cfg.get("mlops_config", {})

    service_name = project_info.get("service_name", default_service_name)
    workflow_type = project_info.get("workflow_type", default_workflow_type)

    tracking_uri = get_tracking_uri(
        repo_root=repo_root,
        tracking_uri=mlops_config.get("tracking_uri", "auto")
    )

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlops_config.get(
        "experiment_name",
        build_experiment_name(
            repo_name="ingeniia_services",
            service_name=service_name,
            workflow_type=workflow_type
        )
    )

    artifact_location = get_artifact_location(
        repo_root=repo_root,
        service_name=service_name,
        workflow_type=workflow_type
    )

    experiment_id = ensure_experiment(
        experiment_name=experiment_name,
        artifact_location=artifact_location
    )

    mlflow.set_experiment(experiment_name)

    run_name = mlops_config.get(
        "run_name",
        f"{project_info.get('experiment_id', 'EXP-000')}_{service_name}_{workflow_type}"
    )

    standard_tags = build_standard_tags(
        cfg=cfg,
        service_name=service_name,
        workflow_type=workflow_type
    )

    return {
        "repo_root": repo_root,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "artifact_location": artifact_location,
        "run_name": run_name,
        "standard_tags": standard_tags,
    }