# 🩻 X-Ray Evaluation API (CNN + GradCAM)

Este microservicio proporciona una API para la clasificación de radiografías de tórax (Normal vs. Anomalía) y la generación de explicabilidad visual (Heatmaps) utilizando una arquitectura Convolutional Neural Network (CNN) basada en YOLOv11.

Este proyecto es parte del curso de Deep Learning de la plataforma [inGeniia.co](https://www.ingeniia.co), donde enseñamos a desplegar modelos de visión por computadora en producción.

---

## 📋 Descripción

El servicio implementa un pipeline completo de visión artificial. Recibe una imagen en formato Base64, la preprocesa y ejecuta dos tareas simultáneas:

1. **Clasificación:** Utiliza `YOLOv11-cls` para determinar si la radiografía presenta anomalías.
2. **Explicabilidad:** Utiliza algoritmos de `GradCAM` para generar un mapa de calor que resalta las regiones donde el modelo "miró" para tomar su decisión.

### Características Principales

- **SOTA Vision Model:** Utiliza la arquitectura `YOLOv11` de [Ultralytics](https://github.com/ultralytics/ultralytics), optimizada para velocidad y precisión.

- **Explicabilidad (XAI):** Generación automática de mapas de calor (Heatmaps) y superposiciones (Overlays) para auditoría médica.

- **Alta Performance:** Pipeline optimizado con PyTorch y OpenCV, con medición detallada de latencia.

- **Validación Estricta:** Esquemas Pydantic para validar la integridad de la imagen Base64.

- **Dockerizado:** Entorno reproducible basado en Python 3.11 Slim con soporte para librerías gráficas.

---

## 🏁 Guía de Construcción

### Paso 1: Preparación de Artefactos

- Asegúrate de que el modelo entrenado .pt se encuentre en la ruta correcta:
`src/models/YOLO/xrays_evaluation_model_medium_v1.pt`

### Paso 2: Construcción de la Imagen Docker

- Desde el directorio raíz del repositorio (ingeniia_services/), ejecuta:

  ```bash
  docker build -t genia/xrays-evaluation-cnn:1.0 -f container-images/xrays_evaluation/Dockerfile .
  ```

### Paso 3: Ejecutar el Contenedor

- Una vez construida la imagen:

  ```bash
  docker run -d -p 8080:8080 --name xrays-service genia/xrays-evaluation-cnn:1.0
  ```

### Paso 4: Verificar Funcionamiento

- Accede a la documentación interactiva:
`http://localhost:8080/docs`

---

## 📝 Cómo Usar la API

El endpoint principal es /cnn_xray_demo.

- Ejemplo de Solicitud (cURL)
  - Debes enviar la imagen codificada en Base64.

    ```bash
    curl -X 'POST' \
      'http://localhost:8080/cnn_xray_demo' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
    }'
    ```


- Respuesta Exitosa Esperada

  - El servicio retorna la predicción, la explicabilidad (imagen overlay en base64) y los tiempos de ejecución.

    ```bash
    {
      "prediction": {
        "label": "Anomaly",
        "confidence": 0.985,
        "class_id": 0
      },
      "explainability": {
        "heatmap_base64": "...",
        "overlay_base64": "...",
        "description": "Red indicates high attention regions."
      },
      "performance": {
        "preprocess_time_ms": 12.5,
        "inference_time_ms": 15.2,
        "explainability_time_ms": 8.1,
        "total_latency_ms": 35.8,
        "model_used": "YOLO11m-cls"
      }
    ```

---

## ⚙️ Gestión del Contenedor

Comandos útiles para el ciclo de vida del servicio:

```bash
# Detener el servicio
docker stop xrays-evaluation-service

# Ver logs (útil para errores de decodificación de imagen)
docker logs -f xrays-evaluation-service

# Eliminar el contenedor
docker rm xrays-evaluation-service
```

---

## ☁️ Despliegue en Google Cloud Platform (GCP)

Debido a la naturaleza pesada de las redes neuronales convolucionales, este servicio requiere una configuración de hardware superior en Cloud Run.

### Configuración del Despliegue (Cloud Build)

El archivo de despliegue `cloudbuild-xrays_evaluation_service.yaml` está configurado con recursos aumentados:
- **CPU:** `2 vCPU`.
- **Memoria:** `8Gi`.
- **Región:** `us-central1`.
- **CPU:** `--allow-unauthenticated`

---

## 📄 Licencia y Atribución

Este proyecto se distribuye bajo la licencia GNU Affero General Public License v3.0 (AGPL-3.0).

### Atribución a inGeniia.co
Copyright (c) 2024 inGeniia.co Si utilizas este código con fines educativos o comerciales, debes incluir la siguiente mención:

```text
Este software implementa arquitecturas de Deep Learning basadas en los materiales educativos de inGeniia.co.
```

### Atribución a Ultralytics YOLO

Este proyecto utiliza código y modelos de arquitectura basados en Ultralytics YOLO, licenciado bajo la AGPL-3.0. Puedes encontrar el repositorio original en [Ultralytics](https://github.com/ultralytics/ultralytics).

Para conocer los detalles y condiciones de distribución consulte el archivo [LICENCIA.](https://github.com/AprendeIngenia/deep_learning_services/blob/a01e33ae3f686225fc5e3d322b3979fc90120202/python/xrays_evaluation/LICENSE)

---

**¿Te interesa aprender más?** 

Visita [www.inGeniia.co](https://www.ingeniia.co) para acceder al código fuente completo, los videos explicativos y los mapas mentales de esta y otras redes neuronales.



