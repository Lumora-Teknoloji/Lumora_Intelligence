# api/routes/alerts.py
"""GET /alerts — Trend uyarıları listesi."""
from fastapi import APIRouter, Query

from services.intelligence_service import intelligence_service

router = APIRouter()


@router.get("/alerts", tags=["Alerts"])
async def get_alerts(
    unread_only: bool = Query(False, description="Sadece okunmamış alertleri getir"),
    limit:       int  = Query(50,    ge=1, le=200, description="Maksimum kayıt sayısı"),
):
    """
    Trend uyarıları listesi.

    Alert tipleri: `rank_spike`, `rank_drop`, `viral_start`, `category_heat`,
    `feedback_penalty`, `new_brand_entry`
    """
    alerts = intelligence_service.get_alerts(unread_only=unread_only)
    alerts = alerts[:limit]

    return {
        "count":  len(alerts),
        "alerts": alerts,
    }
