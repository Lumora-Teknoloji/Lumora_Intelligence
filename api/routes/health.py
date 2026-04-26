# api/routes/health.py
"""GET /health — Servis ve engine sağlık durumu."""
from fastapi import APIRouter
from datetime import datetime, timezone

from db.connection import check_connection
from services.intelligence_service import intelligence_service

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Servis ve engine sağlık durumu.

    - `status`: ok / degraded / error
    - `engine_trained`: CatBoost eğitildi mi?
    - `db_connected`: DB bağlantısı var mı?
    """
    db_ok = check_connection()
    engine_status = intelligence_service.get_status()

    overall = "ok" if db_ok else "degraded"

    return {
        "status":       overall,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "db_connected": db_ok,
        **engine_status,
    }
