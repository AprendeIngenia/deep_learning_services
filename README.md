# 🚀 Guía de Despliegue Automatizado con Cloud Build y Cloud Run

Este documento detalla el proceso de configuración de un pipeline de Integración Continua y Despliegue Continuo (CI/CD) para los microservicios de este repositorio, utilizando herramientas de Google Cloud Platform.
El objetivo es automatizar completamente el proceso desde que un desarrollador envía código a GitHub hasta que los servicios están actualizados y corriendo en Cloud Run.

## 🎯 Arquitectura del Pipeline

El flujo de trabajo es el siguiente:

- [Push a GitHub] ➔ Stage envía cambios a la rama main.

- [Disparador de Cloud Build] ➔ El push activa un disparador configurado en GCP.

- [Ejecución de cloudbuild.yaml] ➔ Cloud Build ejecuta las instrucciones definidas en el archivo cloudbuild.yaml.

- Construcción: Se construyen las imágenes Docker para cada microservicio.

- Envío: Las imágenes se suben y versionan en Artifact Registry.

- Despliegue: Se le ordena a Cloud Run que se actualice con las nuevas imágenes.

- [Servicios Activos en Cloud Run] ➔ Los microservicios se actualizan y están listos para recibir tráfico.

## 🏁 Guía de Configuración Inicial (Desde Cero)

Esta guía asume que se está configurando el entorno por primera vez.

### 1. Crear la Infraestructura Base en GCP 🏗️

Primero, creamos los contenedores y las identidades que usaremos.

- Crear el Repositorio de Artefactos: Aquí es donde guardaremos nuestras imágenes Docker.

    ```bash
    gcloud artifacts repositories create ingeniia-services \
    --repository-format=docker \
    --location=us-central1 \
    --description="Repositorio para microservicios"
    ```

-  Crear las Cuentas de Servicio: Necesitamos tres identidades distintas para diferentes roles.
    - El Constructor `github-actions-deployer`: La cuenta que Cloud Build usará para ejecutar todo el pipeline.

        ```bash
        gcloud iam service-accounts create github-actions-deployer \
        --display-name="Service Account for Cloud Build CI/CD"
        ```
    - Las Aplicaciones `credit-scoring-sa` y `content-service-sa`: Las identidades que tendrán los servicios una vez que estén corriendo en Cloud Run.
        ```bash
        gcloud iam service-accounts create credit-scoring-sa \
        --display-name="Service Account for Credit Scoring Service"

        gcloud iam service-accounts create content-service-sa \
        --display-name="Service Account for Content Service"
        ```
### 2. Configuración de Permisos (IAM) 🔑

Esta es la parte más crítica. Cada "actor" en nuestro proceso necesita permisos específicos para hacer su trabajo y nada más (principio de menor privilegio).

- A. Permisos para el Constructor `github-actions-deployer`.

    Esta cuenta es la que orquesta todo, por lo que necesita varios permisos:
    - Para subir imágenes a Artifact Registry `roles/artifactregistry.writer`:

        ```bash
        gcloud projects add-iam-policy-binding ingeniiaservices \
        --member="serviceAccount:github-actions-deployer@ingeniiaservices.iam.gserviceaccount.com" \
        --role="roles/artifactregistry.writer"
        ```
    - Para desplegar y administrar servicios en Cloud Run `roles/run.admin`:

        ```bash
        gcloud projects add-iam-policy-binding ingeniiaservices \
        --member="serviceAccount:github-actions-deployer@ingeniiaservices.iam.gserviceaccount.com" \
        --role="roles/run.admin"
        ```
    - Para escribir los registros de la compilación en Cloud Logging `roles/logging.logWriter`:

        ```bash
        gcloud projects add-iam-policy-binding ingeniiaservices \
        --member="serviceAccount:github-actions-deployer@ingeniiaservices.iam.gserviceaccount.com" \
        --role="roles/logging.logWriter"
        ```
    
    - Para poder asignar las otras cuentas de servicio a los servicios de Cloud Run `roles/iam.serviceAccountUser`:

        ```bash
        # Permiso para usar credit-scoring-sa
        gcloud iam service-accounts add-iam-policy-binding \
        credit-scoring-sa@ingeniiaservices.iam.gserviceaccount.com \
        --member="serviceAccount:github-actions-deployer@ingeniiaservices.iam.gserviceaccount.com" \
        --role="roles/iam.serviceAccountUser"

        # Permiso para usar content-service-sa
        gcloud iam service-accounts add-iam-policy-binding \
        content-service-sa@ingeniiaservices.iam.gserviceaccount.com \
        --member="serviceAccount:github-actions-deployer@ingeniiaservices.iam.gserviceaccount.com" \
        --role="roles/iam.serviceAccountUser"
        ```

- B. Permisos para las Aplicaciones `credit-scoring-sa` y `content-service-sa`.

    Estas cuentas solo necesitan un permiso: poder leer (descargar) su propia imagen de contenedor desde Artifact Registry cuando Cloud Run las inicia.
    - Permiso de lectura sobre el repositorio específico:

        ```bash
        gcloud artifacts repositories add-iam-policy-binding ingeniia-services \
        --location=us-central1 \
        --member="serviceAccount:credit-scoring-sa@ingeniiaservices.iam.gserviceaccount.com" \
        --role="roles/artifactregistry.reader"

        gcloud artifacts repositories add-iam-policy-binding ingeniia-services \
        --location=us-central1 \
        --member="serviceAccount:content-service-sa@ingeniiaservices.iam.gserviceaccount.com" \
        --role="roles/artifactregistry.reader"
        ```

### 3. Adaptación de los Contenedores para Cloud Run 🐳

Para que un contenedor funcione en Cloud Run, debe cumplir dos contratos:

1. El Contrato de Puerto: El contenedor debe escuchar en el puerto que Cloud Run le proporciona a través de la variable de entorno `$PORT` (usualmente 8080).
    - *Acción:* Modificamos la última línea CMD en ambos Dockerfile (en container-images/) para usar `$PORT` en lugar de un puerto fijo.
    - *Linea final:* `CMD uvicorn src.server.app:app --host 0.0.0.0 --port $PORT`
2. El Contrato de Recursos: El contenedor debe operar dentro de los límites de memoria y CPU asignados.
    - *Problema:* Nuestros servicios de ML (credit-scoring-service) usan librerías pesadas como torch y pandas, que exceden el límite de memoria por defecto de Cloud Run (512 MiB).
    - *Acción:*En el `cloudbuild.yaml`, añadimos el flag `--memory=1Gi` al comando de despliegue para aumentar la memoria asignada.

### 4. El Orquestador `cloudbuild.yaml` 📜
Este archivo es el corazón del pipeline. Le dice a Cloud Build qué hacer, paso a paso.

```yaml
steps:
# Paso 1: Construir la imagen para credit-scoring-service
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'us-central1-docker.pkg.dev/${PROJECT_ID}/ingeniia-services/credit-scoring-service:${SHORT_SHA}', '--file=container-images/credit_scoring/Dockerfile', '.']

# Paso 2: Enviar (push) la imagen de credit-scoring-service
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'us-central1-docker.pkg.dev/${PROJECT_ID}/ingeniia-services/credit-scoring-service:${SHORT_SHA}']

# Paso 3: Construir la imagen para content-service
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'us-central1-docker.pkg.dev/${PROJECT_ID}/ingeniia-services/content-service:${SHORT_SHA}', '--file=container-images/content_service/Dockerfile', '.']

# Paso 4: Enviar (push) la imagen de content-service
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'us-central1-docker.pkg.dev/${PROJECT_ID}/ingeniia-services/content-service:${SHORT_SHA}']

# Paso 5: Desplegar credit-scoring-service en Cloud Run
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
    - 'run'
    - 'deploy'
    - 'credit-scoring-service'
    - '--image=us-central1-docker.pkg.dev/${PROJECT_ID}/ingeniia-services/credit-scoring-service:${SHORT_SHA}'
    - '--platform=managed'
    - '--region=us-central1'
    - '--ingress=internal'
    - '--service-account=credit-scoring-sa@${PROJECT_ID}.iam.gserviceaccount.com'
    - '--allow-unauthenticated'
    - '--memory=1Gi'

# Paso 6: Desplegar content-service en Cloud Run
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
    - 'run'
    - 'deploy'
    - 'content-service'
    - '--image=us-central1-docker.pkg.dev/${PROJECT_ID}/ingeniia-services/content-service:${SHORT_SHA}'
    - '--platform=managed'
    - '--region=us-central1'
    - '--ingress=internal'
    - '--service-account=content-service-sa@${PROJECT_ID}.iam.gserviceaccount.com'
    - '--allow-unauthenticated'
    - '--memory=1Gi'

options:
  diskSizeGb: 100
  logging: CLOUD_LOGGING_ONLY
```

### 5. Conexión con GitHub (El Disparador) 🔗
El último paso es conectar todo con nuestro repositorio.

1. Ve a la consola de Cloud Build > Activadores.
2. Haz clic en `Crear activador`.
3. Conecta tu repositorio de GitHub y configura el evento `ej: Enviar una rama, main`.
4. En "Configuración", selecciona "Archivo de configuración de Cloud Build" y asegúrate de que apunte a `cloudbuild.yaml`.
5. Paso Crítico: Haz clic en `MOSTRAR OPCIONES AVANZADAS` y en la sección `Cuenta de servicio`, selecciona la cuenta constructora que creamos: `github-actions-deployer@....` Esto le da a Cloud Build la identidad y los permisos correctos para trabajar.
6. Crea el disparador.

¡Y listo! Con esta configuración, cada `push` a `main` resultará en un despliegue automático, seguro y versionado de tus microservicios.