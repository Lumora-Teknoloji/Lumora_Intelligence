# api/routes/research.py
"""
Research API — Tavily araştırma endpoint'leri.

Backend bu endpoint'leri çağırarak araştırma verisi alır.
Intelligence tüm araştırma sonuçlarını DB verileriyle zenginleştirir.
"""
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.tavily_service import tavily_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research"])


# ─── Request / Response Modelleri ─────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Araştırma konusu", min_length=2)
    category: Optional[str] = Field(None, description="Kategori filtresi")


class VisualSearchRequest(BaseModel):
    query: str = Field(..., description="Görsel arama sorgusu")


class ContextSearchRequest(BaseModel):
    query: str = Field(..., description="Bağlam arama sorgusu")
    max_results: int = Field(3, ge=1, le=10)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/market")
async def market_research(req: ResearchRequest):
    """
    Pazar araştırması + DB trend verileri.

    Tavily'den pazar verileri çeker, Intelligence DB'deki
    trend skorları ve category heat ile zenginleştirir.
    """
    logger.info(f"🔍 Market research: {req.topic} (category={req.category})")
    result = tavily_service.market_research(req.topic, category=req.category)
    return {"status": "ok", "data": result}


@router.post("/runway")
async def runway_analysis(req: ResearchRequest):
    """
    Podyum/defile analizi + görsel toplama.

    Vogue, Milan, Paris defile verilerini ve
    runway görsellerini döndürür.
    """
    logger.info(f"👠 Runway analysis: {req.topic}")
    result = tavily_service.runway_analysis(req.topic)
    return {"status": "ok", "data": result}


@router.post("/visual")
async def visual_search(req: VisualSearchRequest):
    """
    Ürün görseli arama.

    Tavily image search ile kaliteli moda görselleri
    bulur ve filtreler.
    """
    logger.info(f"📸 Visual search: {req.query}")
    result = tavily_service.visual_search(req.query)
    return {"status": "ok", "data": result}


@router.post("/context")
async def context_search(req: ContextSearchRequest):
    """
    Kısa bağlam arama.

    Intent analizi sırasında ek bağlam bilgisi sağlar.
    Backend'in intent.py modülü tarafından çağrılır.
    """
    logger.info(f"📖 Context search: {req.query}")
    result = tavily_service.context_search(req.query, max_results=req.max_results)
    return {"status": "ok", "data": result}


@router.post("/comprehensive")
async def comprehensive_research(req: ResearchRequest):
    """
    Kapsamlı araştırma — tüm kaynakları paralel çalıştırır.

    Market + Runway + DB trend verilerini paralel toplar
    ve birleşik bir context döndürür.
    Backend bu birleşik veriyi GPT-4o'ya gönderir.
    """
    logger.info(f"🌐 Comprehensive research: {req.topic} (category={req.category})")
    result = await tavily_service.comprehensive_research(
        req.topic, category=req.category
    )
    return {"status": "ok", "data": result}


@router.get("/status")
async def research_status():
    """Tavily servis durumu."""
    return {"status": "ok", "data": tavily_service.status()}
