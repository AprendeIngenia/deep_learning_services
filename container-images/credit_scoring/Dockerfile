# --- Etapa 1: Build ---
# Usamos una imagen base de Python slim para reducir el tamaño.
FROM python:3.11-slim as builder

# Variables de entorno para optimizar
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiamos solo el archivo de requerimientos primero para aprovechar el cache de Docker
COPY container-images/credit_scoring/requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Etapa 2: Final ---
# Creamos la imagen final a partir de una base limpia
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copiamos las dependencias instaladas de la etapa de build
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiamos el código fuente del proyecto de credit_scoring
COPY python/credit_scoring/ .

# Exponemos el puerto en el que correrá la aplicación
EXPOSE 8080

# Comando para ejecutar el servidor de Uvicorn cuando el contenedor inicie
CMD ["bash","-lc","uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT:-8080}"]