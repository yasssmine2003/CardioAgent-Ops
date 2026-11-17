# ---------- Étape 1 : construction (installe les dépendances) ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ---------- Étape 2 : image finale (légère, sans outils de compilation) ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba
RUN useradd --create-home app
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY models/cardio_model.onnx models/cardio_model.json models/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
