"""
predictor.py: this class is responsible for loading the artifacts and making the prediction.
"""

import os
import joblib
import torch
import pandas as pd
import logging as log
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.model import CreditScoringModel
from server.schemas import CreditRiskInput

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CreditRiskPredictor:
    """
    Orchestrates the loading of artifacts and the execution of inference.
    """
    def __init__(self, model_path: Path, preprocessor_path: Path, model_config: Dict[str, Any]):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_config = model_config
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        """
        Load the artifacts from the path.
        """
        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
            log.info(f"✔ Archivo preprocesador cargado desde: {self.preprocessor_path}")
        except FileNotFoundError:
            log.error(f"✘ Archivo preprocesador no encontrado en {self.preprocessor_path}")
            raise
        
        try:
            # Recreate the architecture of the model
            self.model = CreditScoringModel(
                num_features=self.model_config['num_features'], # Este valor debe ser el correcto post-procesamiento
                hidden_layers=self.model_config['hidden_layers'],
                dropout_rate=self.model_config['dropout_rate'],
                use_batch_norm=self.model_config['use_batch_norm'],
                activation_fn=self.model_config['activation_fn']
            )
            # load weights trained
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()  # mode: eval
            log.info(f"✔ Pesos del modelo cargados desde: {self.model_path}")
            log.info("✔ Modelo y preprocesador cargados exitosamente.")
        except FileNotFoundError:
            log.error(f"✘ Archivo de modelo no encontrado en {self.model_path}")
            raise
        except Exception as e:
            log.error(f"✘ Error al cargar el modelo: {e}")
            raise
        
    def predict(self, input_data: CreditRiskInput) -> Dict[str, Any]:
        """
        Make the prediction.
        """
        # 1. convert Pydantic input to DataFrame
        input_df = pd.DataFrame([input_data.dict(by_alias=True)])
        
        # 2. apply preprocessing
        processed_features = self.preprocessor.transform(input_df)
        
        # 3. convert to Pytorch tensor
        input_tensor = torch.tensor(processed_features, dtype=torch.float32)
        
        # 4. make prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
            
        # 5. format outout
        prediction = 'good' if probability >= 0.5 else 'bad'
        log.info(f"✔ Predicción generada: {prediction} con probabilidad: {probability:.4f}")
        
        return {
            "prediction": prediction,
            "probability": probability
        }
        

BEST_MODEL_CONFIG = {
    'num_features': 26, 
    'hidden_layers': [256, 128, 64, 64],
    'dropout_rate': 0.1,
    'use_batch_norm': True,
    'activation_fn': 'ReLU'
}

# Paths relativos al root del proyecto `python/credit_scoring`
MODEL_PATH = Path("models/genia_services_mlp_credit_scoring_model_v1.3.0_20250824.pt")
PREPROCESSOR_PATH = Path("models/german_credit_risk_preprocessor.joblib")

# Instancia única (Singleton) del predictor para ser usada por la API.
# Esto asegura que el modelo se carga una sola vez al iniciar el servidor.
predictor_instance = CreditRiskPredictor(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    model_config=BEST_MODEL_CONFIG
)