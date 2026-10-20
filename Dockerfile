# ==============================================================================
# Dockerfile para Producción: Visualizador del Polinomio de Taylor 3D
# ==============================================================================

# Imagen base ligera y oficial de Python
FROM python:3.11-slim AS base

# Variables de entorno para optimización de Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

# Instalar curl para el healthcheck del contenedor y limpiar caché de apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario sin privilegios para ejecución segura
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la aplicación
COPY --chown=appuser:appgroup core/ ./core/
COPY --chown=appuser:appgroup ui/ ./ui/
COPY --chown=appuser:appgroup .streamlit/ ./.streamlit/
COPY --chown=appuser:appgroup app.py .

# Cambiar al usuario no privilegiado
USER appuser

# Exponer el puerto de Streamlit
EXPOSE 8501

# Healthcheck para verificar disponibilidad del servicio
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicio del servidor Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
