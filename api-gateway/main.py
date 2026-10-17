from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="API Gateway Microservice", version="1.0.0")

# Servir les fichiers statiques (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "UP", "service": "api-gateway"}