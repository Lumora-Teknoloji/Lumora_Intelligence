# app.py
"""
Lumora Intelligence — FastAPI Mikro Servis Uygulaması (:8001)

Endpoint'ler:
  GET  /health
  GET  /predict
  POST /analyze
  POST /feedback
  POST /trigger
  GET  /alerts
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from services.intelligence_service import intelligence_service
from services.scheduler_service import create_scheduler
from api.routes import health, predict, analyze, feedback, trigger, alerts, research

logger = logging.getLogger(__name__)


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlatma ve kapatma işlemleri."""
    # ─── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("🚀 Lumora Intelligence başlatılıyor...")

    # DB'den ilk eğitim
    await intelligence_service.startup_train()

    # Scheduler başlat
    scheduler = create_scheduler(intelligence_service)
    scheduler.start()
    logger.info("✅ Scheduler başlatıldı")

    yield  # Uygulama çalışıyor

    # ─── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("🛑 Lumora Intelligence kapatılıyor...")
    scheduler.shutdown(wait=False)


# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lumora Intelligence API",
    description=(
        "Trendyol trend tahmin motoru — 8 algoritma, 6 katman.\n\n"
        "**İç API:** LangChain Backend tarafından `X-Internal-Key` header ile çağrılır."
    ),
    version="1.0.0",
    docs_url="/docs" if config.APP_ENV == "development" else None,
    redoc_url="/redoc" if config.APP_ENV == "development" else None,
    lifespan=lifespan,
)


# ─── Middleware ────────────────────────────────────────────────────────────────

# İç API Key doğrulaması
@app.middleware("http")
async def verify_internal_key(request: Request, call_next):
    """
    /health endpoint'i herkese açık.
    Diğer tüm endpoint'ler X-Internal-Key kontrolü yapar.
    """
    # Health her zaman açık
    if request.url.path == "/health":
        return await call_next(request)

    # Key boşsa (dev mode) → pas geç
    if config.APP_ENV != "production" and (not config.INTERNAL_API_KEY or config.INTERNAL_API_KEY == "lumora-internal-dev-key"):
        return await call_next(request)

    key = request.headers.get("X-Internal-Key", "")
    if key != config.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Geçersiz iç API anahtarı")

    return await call_next(request)


# CORS — sadece LangChain backend'e izin ver
# BACKEND_URL .env'den okunur → VPS'de gerçek domain girilebilir
_cors_origins = [
    config.BACKEND_URL,
    config.BACKEND_URL.replace("http://", "http://").replace("localhost", "127.0.0.1"),
    "http://localhost:8000",       # local dev fallback
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(_cors_origins)),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(analyze.router)
app.include_router(feedback.router)
app.include_router(trigger.router)
app.include_router(alerts.router)
app.include_router(research.router)
