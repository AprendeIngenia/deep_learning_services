import os
import sys
import json
import torch
import torch.nn as nn
import pytest
import logging as log
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


try:
    from src.training.model import CreditScoringModel
except ImportError:
    log.error("Asegúrate de que la estructura de tu proyecto sea correcta.")
    log.error("Este script de test espera estar en una carpeta 'tests/' y el modelo en 'src/training/model.py'")
    sys.exit(1)
    
    
def setup_logging(level=log.INFO, log_file: str | None = None):
    handlers = [log.StreamHandler(sys.stdout)]  # imprime en consola
    if log_file:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))

    log.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    # opcional: bajar ruido de libs terceras
    for noisy in ("mlflow", "urllib3", "matplotlib"):
        log.getLogger(noisy).setLevel(log.WARNING)


setup_logging()


# 1. config
@pytest.fixture(scope="module")
def model_config_fixture():
    """Proporciona una configuración base para el modelo en los tests."""
    return {
        "num_features": 25,
        "hidden_layers": [128, 64],
        "dropout_rate": 0.2,
        "use_batch_norm": True,
        "activation_fn": "ReLU"
    }


@pytest.fixture(scope="module")
def expected_info_fixture():
    """Carga la estructura JSON esperada desde un archivo."""
    json_path = Path(__file__).parent / "expected_model_info.json"
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"El archivo JSON esperado no se encontró en: {json_path}")
    except json.JSONDecodeError:
        pytest.fail(f"Error al decodificar el archivo JSON: {json_path}")
        
        
# 2. tests
def test_model_instantiation(model_config_fixture):
    """
    Verifica que el modelo se puede instanciar sin errores con una configuración válida.
    """
    log.info("TEST: Verificando la instanciación del modelo.")
    try:
        model = CreditScoringModel(**model_config_fixture)
        assert model is not None, "El modelo no debería ser None."
        assert isinstance(model, nn.Module), "El modelo debe ser una instancia de torch.nn.Module."
        log.info("✔ ¡Éxito! El modelo se instanció correctamente.")
    except Exception as e:
        pytest.fail(f"La instanciación del modelo falló con el error: {e}")


def test_model_architecture(model_config_fixture):
    """
    Valida que la arquitectura de la red (capas, dimensiones, orden) se construya correctamente.
    """
    log.info("TEST: Verificando la arquitectura de la red neuronal.")
    model = CreditScoringModel(**model_config_fixture)
    net = model.network
    
    # La arquitectura esperada es: (Linear -> BatchNorm -> ReLU -> Dropout) -> (Linear -> BatchNorm -> ReLU -> Dropout) -> Linear
    expected_layers_count = len(model_config_fixture["hidden_layers"]) * 4 + 1
    assert len(net) == expected_layers_count, f"Se esperaban {expected_layers_count} capas, pero se encontraron {len(net)}."
    
    # 1° hidden layer
    assert isinstance(net[0], nn.Linear) and net[0].in_features == 25 and net[0].out_features == 128
    assert isinstance(net[1], nn.BatchNorm1d) and net[1].num_features == 128
    assert isinstance(net[2], nn.ReLU)
    assert isinstance(net[3], nn.Dropout) and net[3].p == 0.2
    
    # 2° hidden layer
    assert isinstance(net[4], nn.Linear) and net[4].in_features == 128 and net[4].out_features == 64
    assert isinstance(net[5], nn.BatchNorm1d) and net[5].num_features == 64
    assert isinstance(net[6], nn.ReLU)
    assert isinstance(net[7], nn.Dropout) and net[7].p == 0.2
    
    # 3° output layer
    assert isinstance(net[8], nn.Linear) and net[8].in_features == 64 and net[8].out_features == 1
    
    log.info("✔ ¡Éxito! La arquitectura y las dimensiones de las capas son correctas.")


def test_forward_pass(model_config_fixture):
    """
    Realiza un "smoke test" para asegurar que el forward pass se ejecuta y devuelve un tensor con la forma correcta.
    """
    log.info("TEST: Verificando el forward pass.")
    model = CreditScoringModel(**model_config_fixture)
    model.eval()
    
    batch_size = 10
    input_tensor = torch.randn(batch_size, model_config_fixture["num_features"])
    
    with torch.no_grad():
        output = model(input_tensor)
    
    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, f"La forma del tensor de salida es incorrecta. Esperado: {expected_shape}, Obtenido: {output.shape}"
    log.info(f"✔ ¡Éxito! El forward pass se completó y la forma de salida es correcta: {output.shape}.")

