# --- Etapa 1: Builder ---
# Usamos una imagen base de Python slim para reducir el tamaño.
FROM python:3.11-slim as builder

# Variables de entorno para optimizar
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copiamos el archivo de requerimientos desde su ubicación original
COPY requirements.txt .

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

# Copiamos el código fuente. En desarrollo, esto se sobrescribirá con un volumen.
COPY . .

# Exponemos el puerto para el servicio
EXPOSE 8000

# Comando para ejecutar el servidor con recarga automática para desarrollo
CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]