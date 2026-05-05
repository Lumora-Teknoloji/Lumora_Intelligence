# api/routes/predict.py
"""GET /predict — Kategori trend tahmin listesi."""
from fastapi import APIRouter, Query
from typing import Optional

from services.intelligence_service import intelligence_service

router = APIRouter()


import time

_predict_cache = {}
CACHE_TTL = 900  # 15 dakika

@router.get("/predict", tags=["Prediction"])
async def predict(
    category: Optional[str] = Query(None, description="Filtre kategori (search_term)"),
    top_n:    int            = Query(20,   ge=1, description="Kaç ürün dönsün? (max 1000)"),
):
    """
    Trend tahmin listesi döndürür.

    - **category**: Belirli bir kategori filtresi (ör: `crop`, `tayt`). Boş bırakılırsa tümü.
    - **top_n**: Döndürülecek maksimum ürün sayısı. 1000'den büyük değerler 1000'e kısıtlanır.

    Her ürün için: `product_id`, `trend_label`, `trend_score`, `confidence`
    """
    top_n = min(top_n, 1000)
    
    # Cache lookup
    cache_key = f"{category}_{top_n}"
    now = time.time()
    if cache_key in _predict_cache:
        cached_data, timestamp = _predict_cache[cache_key]
        if now - timestamp < CACHE_TTL:
            return cached_data

    # Cache miss
    results = intelligence_service.predict(category=category, top_n=top_n)
    
    response = {
        "count":    len(results),
        "category": category,
        "results":  results,
    }
    
    # Save to cache
    _predict_cache[cache_key] = (response, now)
    
    return response
