# api/routes/analyze.py
"""POST /analyze — Tekil ürün trend analizi."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.intelligence_service import intelligence_service

router = APIRouter()


class AnalyzeRequest(BaseModel):
    product_id: int = Field(..., description="Analiz edilecek ürün ID'si", gt=0)


import time

_analyze_cache = {}
CACHE_TTL = 900  # 15 dakika

@router.post("/analyze", tags=["Prediction"])
async def analyze(request: AnalyzeRequest):
    """
    Tek bir ürün için detaylı trend analizi.

    Döner: `trend_label`, `trend_score`, `confidence`, `signals`, `data_points`
    """
    now = time.time()
    cache_key = request.product_id
    
    if cache_key in _analyze_cache:
        cached_data, timestamp = _analyze_cache[cache_key]
        if now - timestamp < CACHE_TTL:
            return cached_data

    result = intelligence_service.analyze(request.product_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    _analyze_cache[cache_key] = (result, now)
    return result
