from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ProductInfo(BaseModel):
    sku: str
    name: str
    price: float
    currency: str
    description: str

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_norm: List[float] # normalizados
    product_info: Optional[ProductInfo] = None

class CheckoutResponse(BaseModel):
    frame_id: int
    server_time: int
    inference_time: float
    img_size: List[int]
    detections: List[Detection]
    extra: Dict[str, Any] = {}