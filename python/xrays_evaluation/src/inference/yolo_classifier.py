import torch
import numpy as np

from ultralytics import YOLO
from src.processing.interfaces.main import IModelLoader


class YoloClassifier(IModelLoader):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"🔄 Cargando modelo YOLO desde {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ Modelo cargado correctamente.")

    def predict(self, image: np.ndarray) -> dict:
        """
        return predict.
        Nota: Para GradCAM necesitamos acceso a los ganchos (hooks) internos,
        por lo que retornamos el objeto result completo para el Explainer.
        """
        # Run inference
        results = self.model.predict(image, verbose=False)
        result = results[0]

        # Parse output
        probs = result.probs
        top1_index = probs.top1
        confidence = probs.top1conf.item()
        class_name = result.names[top1_index]

        return {
            "class_id": top1_index,
            "class_name": class_name,
            "confidence": confidence,
            "raw_result": result,
            "internal_model": self.model.model
        }
