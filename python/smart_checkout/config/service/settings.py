from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SC_", env_file=".env", extra="ignore")

    # Core
    service_name: str = "smart_checkout_ws"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Inference
    detection_model: Path = Field(default=Path("models/YOLO/smart_checkout_model_small_v1.pt"))
    stream: bool =  False
    device: str = "cuda:0"
    img_size: int = 640
    confidence_threshold: float = 0.55
    iou_threshold: float = 0.45
    max_det: int = 100
    half: bool = True

    # Catalog
    products_path: Path = Field(default=Path("config/products.yaml"))

    # Limits
    max_frame_bytes: int = 2_500_000  # ~2.5MB (jpeg 640x640 suele ser << esto)


settings = Settings()
