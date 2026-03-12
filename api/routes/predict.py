# api/routes/predict.py
"""GET /predict — Kategori trend tahmin listesi."""
from fastapi import APIRouter, Query
from typing import Optional

from services.intelligence_service import intelligence_service

router = APIRouter()


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
    # Softly clamp — 422 yerine clamp et (istemci çökmez)
    top_n = min(top_n, 1000)
    results = intelligence_service.predict(category=category, top_n=top_n)
    return {
        "count":    len(results),
        "category": category,
        "results":  results,
    }
