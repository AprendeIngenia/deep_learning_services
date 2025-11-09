# 🪙 Microservicio de Scoring de Crédito con MLP y FastAPI

Este proyecto ofrece un microservicio de inferencia listo para producción, empaquetado en Docker. Utiliza un modelo de Perceptrón Multicapa (MLP) entrenado con PyTorch para evaluar el riesgo crediticio de un solicitante en tiempo real.

## 🎯 Características Principales

- API Moderna: Construido con FastAPI, que proporciona alta performance y documentación interactiva automática (Swagger UI).

- Modelo de Deep Learning: Utiliza PyTorch para las predicciones, permitiendo arquitecturas de redes neuronales complejas.

- Listo para Desplegar: Totalmente dockerizado, garantizando un entorno consistente y un despliegue sencillo.

- Preprocesamiento Integrado: El pipeline de preprocesamiento de scikit-learn está integrado, asegurando que los datos de inferencia se traten igual que en el entrenamiento.

## 🏁 Guía de Construcción

### Paso 1: Preparación de Artefactos
- Asegúrate de tener los artefactos del modelo (`.pt`) y el preprocesador (`.joblib`) en la carpeta `python/credit_scoring/models/`.

### Paso 2: Construcción de la Imagen Docker
- Navega al directorio raíz `genia_services/` y ejecuta el siguiente comando para construir la imagen.

```bash
docker build -t genia/credit-scoring-mlp:1.0 -f container-images/credit_scoring/Dockerfile .
```
### Paso 3: Ejecutar el Contenedor Docker
- Una vez construida la imagen, levanta un contenedor con este comando:

```bash
 docker run -d -p 8000:8000 --name credit-scoring-service genia/credit-scoring-mlp:1.0
```

### Paso 4: Verificar el Funcionamiento
- Abre tu navegador web y ve a la siguiente URL para acceder a la documentación interactiva de la API:

```bash
http://localhost:8000/docs
```

## 📝 Cómo Usar la API (¡Haciendo una Predicción!)
El endpoint principal es /mlp_demo. Puedes enviarle una solicitud POST con los datos del solicitante en formato JSON.

- Opción A: Usando la Documentación Interactiva (Swagger)

    - Ve a http://localhost:8000/docs.

    - Despliega el endpoint POST /mlp_demo.

    - Haz clic en el botón "Try it out".

    - Modifica el cuerpo de la solicitud (Request body) con los datos del cliente.

    - Haz clic en "Execute". ¡Verás la respuesta del modelo directamente en la página!

- Opción B: Usando cURL desde la Terminal

    - Abre una terminal y ejecuta el siguiente comando cURL para enviar una solicitud de ejemplo:

        ```bash
        curl -X 'POST' \
        'http://localhost:8000/mlp_demo' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "Age": 35,
        "Sex": "male",
        "Job": 2,
        "Housing": "own",
        "Saving accounts": "little",
        "Checking account": "moderate",
        "Credit amount": 2500,
        "Duration": 24,
        "Purpose": "car"
        }'
        ```

- Respuesta Exitosa Esperada (200 OK):
Si todo va bien, recibirás una respuesta como esta, indicando la predicción (good o bad) y la probabilidad asociada:

    ```bash
    {
    "prediction": "good",
    "probability": 0.7852
    }
    ```


## ⚙️ Gestión del Contenedor
Aquí tienes algunos comandos útiles para administrar el contenedor Docker.

- Puedes detener el contenedor en cualquier momento usando:

    ```bash
    docker stop credit-scoring-service  
    ```

- Ver los logs en tiempo real:
    ```bash
    docker logs -f credit-scoring-service 
    ```

- Reiniciar un contenedor detenido::
    ```bash
    docker start credit-scoring-service 
    ```

- Reiniciar un contenedor detenido::
    ```bash
    docker stop credit-scoring-service && docker rm credit-scoring-service
    ```
