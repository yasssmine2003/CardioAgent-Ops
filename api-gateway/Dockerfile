# Stage 1: Build des dépendances
FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Image finale minimale et sécurisée
FROM python:3.11-slim

WORKDIR /app

# Sécurité : Création d'un utilisateur non-root
RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /opt/venv /opt/venv
COPY main.py .
COPY static ./static

# Donner les droits à l'utilisateur non-root
RUN chown -R appuser:appuser /app
USER appuser

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000

VOLUME ["/app/uploads"]

# Healthcheck natif Docker
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]