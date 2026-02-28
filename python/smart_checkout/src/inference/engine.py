import cv2
import time
import logging as log
import numpy as np

from pathlib import Path
from ultralytics import YOLO

from config.settings import settings
from src.server.schemas import Detection, CheckoutResponse
from src.processing.catalog import ProductCatalog


class ObjectDetector:
    def __init__(self):
        self.model: YOLO | None = None
        self.catalog = ProductCatalog()
        self.device = settings.device

    @staticmethod
    def _resolve_path(p: Path) -> Path:
        base_dir = Path(__file__).resolve().parents[2]
        return (base_dir / p).resolve() if not p.is_absolute() else p

    def load_model(self):
        """Load detection model."""
        model_path = self._resolve_path(settings.detection_model)
        log.info(f"🚀 Cargando modelo desde: {model_path} en {self.device}")

        if not settings.detection_model.exists():
            log.critical(f"❌ El archivo del modelo no existe: {settings.detection_model}")
            raise FileNotFoundError("Modelo no encontrado")

        try:
            self.model = YOLO(str(model_path))
            self.model.to(self.device)

            dummy_input = np.zeros((settings.img_size, settings.img_size, 3), dtype=np.uint8)
            self.model.predict(dummy_input, verbose=False, device=self.device)

            log.info("✅ Modelo cargado y listo (Warmup completo).")
        except Exception as e:
            log.critical(f"❌ Error fatal cargando el modelo: {e}")
            raise e

    def predict(self, image_bytes: bytes, frame_id: int) -> CheckoutResponse:
        start_time = time.time()

        # 1. decode images (Bytes -> Numpy)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 2. inference
        results = self.model.predict(
            frame,
            imgsz=settings.img_size,
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            max_det=settings.max_det,
            device=self.device,
            verbose=False,
            classes=None
        )

        detections_list = []
        result = results[0]

        # 3. mapping
        for box in result.boxes:
            coords = box.xywhn[0].tolist()
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            conf = float(box.conf[0])

            # additional info
            prod_info = self.catalog.get_product_info(class_name)

            detections_list.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=round(conf, 2),
                bbox_norm=coords,
                product_info=prod_info
            ))

        process_time_ms = (time.time() - start_time) * 1000.0

        return CheckoutResponse(
            frame_id=frame_id,
            server_time=int(time.time() * 1000),  # Timestamp actual en ms
            inference_time=round(process_time_ms, 2),
            img_size=[frame.shape[1], frame.shape[0]],  # [width, height] real
            detections=detections_list,
            extra={"total_items": len(detections_list)}
        )