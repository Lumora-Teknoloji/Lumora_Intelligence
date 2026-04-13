# api/routes/feedback.py
"""
Feedback loop API — üretim kararları ve gerçek satış verileri.

Endpoint'ler:
  POST /feedback          → Kalman filter güncelleme (mevcut)
  POST /feedback/decision → Üretim kararı kaydet
  POST /feedback/sale     → Gerçek satış verisi kaydet + Kalman feedback
  GET  /feedback/summary  → Feedback loop durumu
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from services.intelligence_service import intelligence_service
from db.writer import save_production_decision, save_actual_sale, get_feedback_stats
from db.reader import get_production_decisions, get_actual_sales, get_feedback_summary

router = APIRouter()


class FeedbackRequest(BaseModel):
    product_id:          int = Field(..., description="Ürün ID'si", gt=0)
    sold_quantity:       int = Field(..., description="Gerçek satılan adet", ge=0)
    predicted_quantity:  int = Field(..., description="Sistemin tahmin ettiği adet", ge=0)


class DecisionRequest(BaseModel):
    product_id:      int = Field(..., description="Ürün ID'si", gt=0)
    search_term:     str = Field("", description="Kategori (crop, tayt...)")
    predicted_score: Optional[float] = Field(None, description="Intelligence trend_score")
    decision:        str = Field("produce", description="produce / skip / wait")
    quantity:        int = Field(0, description="Üretilecek adet", ge=0)
    notes:           Optional[str] = Field(None, description="Serbest not")


class SaleRequest(BaseModel):
    production_id:     int = Field(..., description="production_decisions.id", gt=0)
    sold_quantity:     int = Field(..., description="Gerçek satılan adet", ge=0)
    produced_quantity: int = Field(..., description="Üretilen adet", ge=0)
    revenue:           Optional[float] = Field(None, description="Gerçek gelir (TL)")


@router.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Gerçek satış verisiyle Kalman filter ve ensemble ağırlıklarını günceller.

    - **penalty_applied**: True ise tahmin %50'den fazla saptı → ağırlık cezası uygulandı
    - **error_pct**: Sapma yüzdesi
    """
    result = intelligence_service.submit_feedback(
        product_id=request.product_id,
        sold_quantity=request.sold_quantity,
        predicted_quantity=request.predicted_quantity,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message"))

    return result


@router.post("/feedback/decision", tags=["Feedback"])
async def record_decision(request: DecisionRequest):
    """
    Üretim kararını kaydeder.

    Intelligence TREND dediği bir ürün için:
    - 'produce' → üretilecek
    - 'skip' → üretilmeyecek
    - 'wait' → bekleniyor
    """
    new_id = save_production_decision(request.model_dump())
    if new_id is None:
        raise HTTPException(status_code=500, detail="Üretim kararı kaydedilemedi")
    return {"status": "ok", "decision_id": new_id}


@router.post("/feedback/sale", tags=["Feedback"])
async def record_sale(request: SaleRequest):
    """
    Gerçek satış verisini kaydeder ve Kalman feedback loop'u tetikler.

    sell_through_rate otomatik hesaplanır:
      1.0 → mükemmel (tamamı satıldı)
      0.0 → hiç satılmamış
    """
    new_id = save_actual_sale(request.model_dump())
    if new_id is None:
        raise HTTPException(status_code=500, detail="Satış verisi kaydedilemedi")

    # Kalman feedback'i de tetikle (aktif ürün için)
    try:
        decisions = get_production_decisions(product_id=None, limit=1)
        # production_id'den product_id'yi bul
        from sqlalchemy import text as sql_text
        from db.connection import engine as db_engine
        with db_engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT product_id FROM production_decisions WHERE id = :pid"),
                {"pid": request.production_id},
            ).fetchone()
            if row and row[0]:
                intelligence_service.submit_feedback(
                    product_id=row[0],
                    sold_quantity=request.sold_quantity,
                    predicted_quantity=request.produced_quantity,
                )
    except Exception:
        pass  # Kalman feedback opsiyonel — hata sessizce loglanır

    return {"status": "ok", "sale_id": new_id}


@router.get("/feedback/summary", tags=["Feedback"])
async def feedback_summary():
    """
    Feedback loop durumunu özetler.

    - **ready_for_catboost**: True ise 30+ satış verisi birikmiş → retraining yapılabilir
    """
    return get_feedback_summary()

