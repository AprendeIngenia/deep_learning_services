import logging as log

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.inference.engine import ObjectDetector
from src.server.router import router as ws_router

log.basicConfig(level=getattr(log, settings.log_level, log.INFO),
               format="%(asctime)s - %(levelname)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.
    Carga el modelo de IA antes de recibir peticiones.
    """
    log.info(f"🔄 Iniciando servicio: {settings.service_name}...")

    # 1. initialize detector
    try:
        detector = ObjectDetector()
        detector.load_model()

        # 2. inject detector
        app.state.detector = detector
        log.info("✅ Motor de inferencia listo e inyectado en app.state.")
    except Exception as e:
        log.critical(f"🔥 Fallo crítico al iniciar el motor de inferencia: {e}")
        raise e

    yield

    # 3. shutdown
    log.info("🛑 Deteniendo servicio...")
    if hasattr(app.state, "detector"):
        del app.state.detector
        log.info("🗑️ Recursos liberados.")


app = FastAPI(
    title="Smart Checkout Detection Service",
    description="Microservicio de detección de objetos en tiempo real (YOLO26) vía WebSockets para inGeniia.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
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


# --- Routing ---
app.include_router(ws_router)


# --- Standard Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    """Redirecciona a la documentación automática."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Endpoint para Health Checks de GCP/K8s.
    Verifica que el modelo esté cargado en memoria.
    """
    is_ready = hasattr(app.state, "detector") and app.state.detector.model is not None

    status = "ok" if is_ready else "not_ready"
    return {
        "status": status,
        "service": settings.service_name,
        "model": settings.detection_model.name,
        "device": settings.device
    }
