"""
python/credit_scoring/src/server/app.py:  uses FastAPI to create the /predict endpoint that receives the data, passes it to the predictor_instance, and returns the result.
"""

import os
import sys
import logging as log

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.server.schemas import CreditRiskInput, CreditRiskOutput
from src.inference.predictor import predictor_instance

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# init FastAPI
app = FastAPI(
    title="API de Inferencia para Scoring de Crédito",
    description="Un microservicio para predecir el riesgo crediticio usando un modelo MLP con PyTorch.",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://www.ingeniia.co"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EndPoint API
@app.get("/", include_in_schema=False)
async def root():
    """Redirige a la documentación de la API."""
    return RedirectResponse(url="/docs")

@app.post("/mlp_demo", 
          response_model=CreditRiskOutput,
          tags=["Predicciones"],
          summary="Realiza una predicción de riesgo crediticio")

async def predict_credit_risk(request: CreditRiskInput) -> CreditRiskOutput:
    """
    Receives applicant data and returns a credit risk prediction.
    
    - Input: JSON with applicant attributes.
    - Output: JSON with the prediction (good or bad) and the associated probability.
    """
    try:
        log.info(f"Recibida solicitud para predicción: {request.dict(by_alias=True)}")
        prediction_result = predictor_instance.predict(request)
        return CreditRiskOutput(**prediction_result)
    except Exception as e:
        log.error(f"Error durante la predicción: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Ocurrió un error interno al procesar la solicitud: {e}"
        )
        

"""
local execute:
uvicorn src.server.app:app --reload
"""