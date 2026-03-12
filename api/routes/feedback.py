# api/routes/feedback.py
"""POST /feedback — Gerçek satış verisiyle feedback loop güncelleme."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.intelligence_service import intelligence_service

router = APIRouter()


class FeedbackRequest(BaseModel):
    product_id:          int = Field(..., description="Ürün ID'si", gt=0)
    sold_quantity:       int = Field(..., description="Gerçek satılan adet", ge=0)
    predicted_quantity:  int = Field(..., description="Sistemin tahmin ettiği adet", ge=0)


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
