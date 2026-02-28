import asyncio
import logging as log

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState

from config.settings import settings
from src.server.protocol import unpack_message

router = APIRouter()
GPU_LOCK = asyncio.Lock()


@router.websocket("/ws/smart_checkout")
async def websocket_inference(websocket: WebSocket):
    """
    Endpoint de detección en tiempo real.
    Protocolo:
    - Cliente envía: [Frame ID (8 bytes uint64)] + [Imagen JPEG (Bytes restantes)].
    - Servidor responde: JSON (CheckoutResponse).
    """
    await websocket.accept()

    detector = websocket.app.state.detector
    client_info = f"{websocket.client.host}:{websocket.client.port}"

    log.info(f"🔌 Cliente conectado: {client_info}")

    try:
        while True:
            # 1. received message
            message = await websocket.receive_bytes()

            # 2. check size
            if len(message) > settings.max_frame_bytes:
                error_msg = {
                    "frame_id": -1,
                    "error": "frame_too_large",
                    "limit": settings.max_frame_bytes
                }
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(error_msg)
                continue

            # 3. unpack message
            try:
                frame_id, jpeg_bytes = unpack_message(message)
            except Exception as e:
                log.warning(f"⚠️ Error de protocolo desde {client_info}: {e}")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"frame_id": -1, "error": f"bad_message: {e}"})
                continue

            # 4. inference
            try:
                async with GPU_LOCK:
                    response = await asyncio.to_thread(detector.predict, jpeg_bytes, frame_id)

                # 5. response
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(response.model_dump())

            except ValueError as ve:
                # decode error
                await websocket.send_json({"frame_id": frame_id, "error": "image_decode_failed"})

            except Exception as e:
                log.error(f"❌ Error de inferencia frame {frame_id}: {e}")
                await websocket.send_json({"frame_id": frame_id, "error": f"inference_error: {e}"})


    except WebSocketDisconnect:
        log.info(f"👋 Cliente desconectado.")

    except Exception as e:
        log.error(f"❌ Error crítico en WebSocket: {e}")
        try:
            await websocket.close(code=1011)  # Internal Error
        except:
            pass