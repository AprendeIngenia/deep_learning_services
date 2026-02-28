import yaml
import logging as log
from pathlib import Path
from typing import Dict, Any

from src.server.schemas import ProductInfo
from config.settings import settings

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ProductCatalog:
    def __init__(self, catalog_path: Path | None = None):
        self._catalog_path = catalog_path or self._resolve_path(settings.products_path)
        self._currency = "USD"
        self._products: Dict[str, Dict[str, Any]] = {}
        self.reload()

    @staticmethod
    def _resolve_path(p: Path) -> Path:
        base_dir = Path(__file__).resolve().parents[2]
        return (base_dir / p).resolve() if not p.is_absolute() else p

    def reload(self) -> None:
        if not self._catalog_path.exists():
            log.warning(f"⚠️ Catálogo no encontrado en {self._catalog_path}. Se usará vacío.")
            self._products = {}
            self._currency = "USD"
            return

        with open(self._catalog_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._currency = data.get("currency", "USD")
        products = data.get("products", {}) or {}

        normalized = {}
        for k, v in products.items():
            key = str(k).lower().strip()
            info = dict(v) if isinstance(v, dict) else {}
            info.setdefault("currency", self._currency)
            normalized[key] = info

        self._products = normalized
        log.info(f"✅ Catálogo cargado: {len(self._products)} productos | currency={self._currency}")

    def get_product_info(self, class_name: str) -> ProductInfo:
        key = str(class_name).lower().strip()
        info = self._products.get(key)

        if info:
            return ProductInfo(**info)

        return ProductInfo(
            sku="UNK-000",
            name=f"Unknown ({class_name})",
            price=0.0,
            currency=self._currency,
            description="Producto no encontrado en catálogo."
        )
