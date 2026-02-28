# 🛒 Smart Checkout API (YOLO Object Detection + WebSockets)

Este microservicio proporciona una API en tiempo real para la detección de múltiples productos (Object Detection) simulando el comportamiento de un Cajero Autónomo de supermercado. Utiliza una arquitectura Convolutional Neural Network (CNN) basada en YOLOv11 y comunicación bidireccional por WebSockets.

Este proyecto es parte del curso de Deep Learning de la plataforma [inGeniia.co](https://www.ingeniia.co), donde enseñamos a desplegar modelos de visión por computadora en infraestructura Cloud acelerada por GPU.

---

## 📋 Descripción

A diferencia de las APIs REST tradicionales, este servicio implementa un pipeline de transmisión de video en tiempo real. Recibe frames de video empaquetados en un protocolo binario, ejecuta la inferencia en una GPU dedicada y devuelve las detecciones junto con la información comercial del producto.

### Características Principales

- **Real-Time Streaming:** Uso de `WebSockets` (FastAPI/Starlette) para comunicación bidireccional con latencia de milisegundos, evitando el sobrecosto de las cabeceras HTTP en cada frame.
- **SOTA Vision Model:** Utiliza la arquitectura `YOLOv11/26` de [Ultralytics](https://docs.ultralytics.com/models/yolo26/?utm_source=infl&utm_medium=twitter&utm_campaign=yolo26&utm_term=ingeniia), optimizada para detección múltiple rápida.
- **Protocolo Binario Eficiente:** El cliente no envía JSON ni Base64, sino un `ArrayBuffer` crudo (`[8 bytes uint64 Frame ID] + [Imagen JPEG]`), optimizando el ancho de banda.
- **Inferencia en GPU (CUDA):** Configurado para ejecutar los tensores directamente en la VRAM (`cuda:0`), gestionado por `asyncio.Lock()` para prevenir condiciones de carrera (Race Conditions) bajo alta concurrencia.
- **Catálogo en Memoria:** Mapeo automático de las clases detectadas con una base de datos YAML (`products.yaml`) para enriquecer la respuesta con precios (USD) y SKUs.

---

## 🏁 Guía de Construcción

Al utilizar dependencias de GPU (CUDA), la construcción de esta imagen requiere configuraciones específicas en el Dockerfile.

### Paso 1: Preparación de Artefactos

Asegúrate de que el modelo entrenado (`.pt`) se encuentre en la ruta correcta especificada en `config/settings.py`:
`models/YOLO/smart_checkout_model_small_v1.pt`

Asegúrate de tener el catálogo de productos configurado en:
`config/products.yaml`

### Paso 2: Construcción de la Imagen Docker

- Desde el directorio raíz del repositorio (`ingeniia_services/`), ejecuta:

    ```bash
    docker build -t genia/smart-checkout-ws:1.0 -f container-images/smart_checkout_service/Dockerfile .
    ```


### Paso 3: Ejecutar el Contenedor

- Una vez construida la imagen:

  ```bash
  docker run -d -p 8080:8080 --name smart-checkout-ws genia/smart-checkout-ws:1.0
  ```
  
### Paso 4: Verificar Funcionamiento

- Puedes verificar el estado de salud del servicio y si el modelo cargó correctamente en memoria visitando: `http://localhost:8080/health`

---

## 📝 Cómo Usar la API (WebSockets)

Puedes probar el servicio ejecutando el ejemplo de cliente web que esta en: `examples/web_client/`

**Protocolo de Mensajería (Cliente -> Servidor)**

El cliente debe enviar un mensaje Binario (ArrayBuffer) con la siguiente estructura exacta:

- Header (8 bytes): Un número entero sin signo de 64 bits (uint64) en formato Little-Endian, que representa el frame_id (usualmente un timestamp).

- Payload (Resto de bytes): Los bytes puros de la imagen comprimida en formato JPEG.

**Respuesta del Servidor (Servidor -> Cliente)**

El servidor responderá con un string en formato JSON validado por los esquemas de Pydantic:
  ```JSON
  {
  "frame_id": 1708453456789,
  "server_time": 1708453456810,
  "inference_time": 15.4,
  "img_size": [640, 640],
  "detections": [
    {
      "class_id": 2,
      "class_name": "manzana roja",
      "confidence": 0.94,
      "bbox_norm": [0.45, 0.60, 0.12, 0.15],
      "product_info": {
        "sku": "FRU-APL-001",
        "name": "Manzana roja",
        "price": 0.35,
        "currency": "USD",
        "description": "Manzana roja fresca"
      }
    }
  ],
  "extra": {
    "total_items": 1
  }
}
  ```

---

## ☁️ Despliegue Serverless en Google Cloud Run (NVIDIA L4 GPU)

Este servicio está diseñado para escalar desde cero (Scale-to-Zero) utilizando hardware acelerado en Google Cloud.

El archivo de despliegue `ops/cloudbuild-smart_checkout_service.yaml` contiene banderas críticas para el aprovisionamiento de GPUs:

- **GPU Type:** --gpu-type=nvidia-l4 (Arquitectura Ada Lovelace, óptima para inferencia rápida).
- **GPU Count:** --gpu=1 
- **Redundancia:** --no-gpu-zonal-redundancy (Requerido en ciertas regiones como us-east4 por disponibilidad de cuota). 
- **CPU / Memoria:** --cpu=4 / --memory=16Gi (Para soportar la caché de CUDA y la concurrencia de WebSockets). 
- **Concurrencia WS:** --concurrency=20 (Permite múltiples streams de video por contenedor).

---

## 📄 Licencia y Atribución

Este proyecto se distribuye bajo la licencia GNU Affero General Public License v3.0 (AGPL-3.0).

Atribución a inGeniia.co
Copyright (c) 2026 inGeniia.co. Si utilizas este código con fines educativos o comerciales, debes incluir la siguiente mención:

  ```Plaintext
  Este software implementa arquitecturas de Deep Learning basadas en los materiales educativos de inGeniia.co.
  ```

**Atribución a Ultralytics YOLO**
Este proyecto utiliza código y modelos de arquitectura basados en Ultralytics YOLO, licenciado bajo la AGPL-3.0. Puedes encontrar el repositorio original en [Ultralytics](https://docs.ultralytics.com/models/yolo26/?utm_source=infl&utm_medium=twitter&utm_campaign=yolo26&utm_term=ingeniia).

Para conocer los detalles y condiciones de distribución consulte el archivo LICENSE.

---

¿Te interesa aprender más? Visita www.inGeniia.co para acceder al curso completo, los videos explicativos donde construimos esta arquitectura paso a paso, y la comunidad de ingenieros de Machine Learning.


