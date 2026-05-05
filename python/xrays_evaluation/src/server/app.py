import os
import sys
import logging as log

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.server.schemas import XRayInput, XRayOutput
from src.processing.runner import XRayInferencePipeline

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# init FastAPI
app = FastAPI(
    title="API de Evaluación de Radiografías (CNN Demo)",
    description="Microservicio educativo para clasificación de radiografías usando YOLOv11 + GradCAM.",
    version="1.0.0"
)

# CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "https://www.ingeniia.co",
    "https://platform.ingeniia.co"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# instance
cnn_xrays_demo = XRayInferencePipeline()


# --- Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    """docs API."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Healthcheck to GCP."""
    return {"status": "ok", "model": "YOLO11m-cls"}


@app.post("/cnn_xray_demo",
          response_model=XRayOutput,
          tags=["Predicciones CNN"],
          summary="Clasifica una radiografía y genera mapa de calor")
async def predict_xray(request: XRayInput) -> XRayOutput:
    """
    :param
        image in Base64
    :return
        prediction (anomaly/normal)
        heatmap
        performance
    """
    try:
        log.info(f"🚀 Recibida solicitud CNN. Tamaño Base64: {len(request.image_base64)}")
        result = cnn_xrays_demo.run(request.image_base64)
        log.info(f"✅ Inferencia exitosa. Resultado: {result['prediction']['label']}")
        return XRayOutput(**result)

    except ValueError as ve:
        log.error(f"Error de validación o decodificación: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        log.error(f"Error crítico durante la inferencia: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ocurrió un error interno al procesar la radiografía: {str(e)}"
        )