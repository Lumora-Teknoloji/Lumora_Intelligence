# services/tavily_service.py
"""
Tavily Research Service — Intelligence'ın ana araştırma/analiz katmanı.

Tavily API sonuçlarını DB'deki trend verileri ve
kategori sinyalleriyle birleştirerek zenginleştirilmiş
araştırma çıktısı üretir.

Kullanım:
    service = TavilyResearchService()
    result  = await service.comprehensive_research("crop top 2026", category="crop")
"""
import asyncio
import logging
from typing import Optional
from functools import lru_cache

import config

logger = logging.getLogger(__name__)

# ─── Tavily Client ────────────────────────────────────────────────────────────

_tavily_client = None


def _get_tavily():
    """Lazy singleton TavilyClient."""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    api_key = config.TAVILY_API_KEY
    if not api_key:
        logger.warning("⚠ TAVILY_API_KEY tanımlı değil — araştırma devre dışı")
        return None

    try:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("✅ Tavily client başlatıldı")
        return _tavily_client
    except Exception as e:
        logger.error(f"❌ Tavily başlatma hatası: {e}")
        return None


# ─── DB Zenginleştirme Yardımcıları ──────────────────────────────────────────

def _get_db_trend_context(category: Optional[str] = None, top_n: int = 10) -> str:
    """Intelligence DB'den trend verilerini çeker — araştırma çıktısına eklenir."""
    try:
        from services.intelligence_service import intelligence_service
        preds = intelligence_service.predict(category=category, top_n=top_n)

        if not preds:
            return ""

        lines = ["### 📊 LUMORA TREND VERİLERİ ###"]
        for p in preds[:top_n]:
            label = p.get("trend_label", "?")
            score = p.get("trend_score", 0)
            name  = p.get("name", f"Ürün #{p.get('product_id', '?')}")
            lines.append(f"- {name}: {label} (skor={score:.1f})")

        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"DB trend context alınamadı: {e}")
        return ""


def _get_category_heat(category: str) -> Optional[float]:
    """Kategori ısı haritasından heat değerini döndürür."""
    try:
        from db.connection import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            sql = text("""
                SELECT category_heat FROM category_daily_signals
                WHERE search_term = :cat
                ORDER BY signal_date DESC LIMIT 1
            """)
            result = conn.execute(sql, {"cat": category}).scalar()
            return float(result) if result else None
    except Exception:
        return None


# ─── Görsel Kalite Filtresi ───────────────────────────────────────────────────

_BLOCKED_DOMAINS = [
    "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
    "pinterest.com/pin", "youtube.com", "svg", "gif", "favicon",
    "logo", "icon", "banner", "avatar",
]


def _is_quality_fashion_image(url: str) -> bool:
    """Kaliteli moda görseli mi kontrolü."""
    if not url or not url.startswith("http"):
        return False
    url_lower = url.lower()
    return not any(blocked in url_lower for blocked in _BLOCKED_DOMAINS)


# ═════════════════════════════════════════════════════════════════════════════
#  TavilyResearchService
# ═════════════════════════════════════════════════════════════════════════════


class TavilyResearchService:
    """
    Intelligence'ın araştırma katmanı.

    Tavily API sonuçlarını DB verisiyle birleştirerek
    stratejik rapor verisi üretir.
    """

    def __init__(self):
        self._max_results = config.TAVILY_MAX_RESULTS
        self._search_depth = config.TAVILY_SEARCH_DEPTH

    # ─── 1. Pazar Araştırması ─────────────────────────────────────────────

    def market_research(self, topic: str, category: str = None) -> dict:
        """
        Pazar araştırması + DB trend zenginleştirme.

        Returns:
            {
                "context": str,       # Tavily + DB birleşik metin
                "sources": [str],     # Kaynak URL'leri
                "trend_data": str,    # DB trend özeti
                "category_heat": float | None,
            }
        """
        client = _get_tavily()
        sources = []
        context = "### PAZAR VERİSİ ###\n"

        if client:
            queries = [
                f"{topic} 2026 trends consumer behavior",
                f"{topic} best sellers market analysis 2025",
            ]
            for q in queries:
                try:
                    res = client.search(
                        query=q,
                        search_depth=self._search_depth,
                        include_images=False,
                        max_results=self._max_results,
                    )
                    for r in res.get("results", []):
                        context += f"BAŞLIK: {r.get('title')}\nİÇERİK: {r.get('content', '')}\n\n"
                        if r.get("url"):
                            sources.append(r["url"])
                except Exception as e:
                    logger.warning(f"Market search hatası: {e}")
        else:
            context += "(Tavily API devre dışı)\n"

        # DB'den trend verileri
        trend_data = _get_db_trend_context(category=category, top_n=10)
        if trend_data:
            context += f"\n{trend_data}\n"

        heat = _get_category_heat(category) if category else None

        return {
            "context": context,
            "sources": sources,
            "trend_data": trend_data,
            "category_heat": heat,
        }

    # ─── 2. Podyum / Defile Analizi ───────────────────────────────────────

    def runway_analysis(self, topic: str) -> dict:
        """
        Podyum araştırması + görsel toplama.

        Returns:
            {
                "context": str,
                "runway_images": [str],
                "sources": [str],
            }
        """
        client = _get_tavily()
        context = "### RUNWAY VERİSİ ###\n"
        raw_images = []
        sources = []

        if not client:
            return {"context": "(Tavily API devre dışı)", "runway_images": [], "sources": []}

        queries = [
            f"Vogue Runway {topic} trends Spring/Summer 2026 Paris Milan -buy",
            f"high fashion designer collections 2025 {topic} catwalk photos",
        ]

        for q in queries:
            try:
                res = client.search(
                    query=q,
                    search_depth="advanced",
                    include_images=True,
                    max_results=self._max_results,
                )
                for r in res.get("results", []):
                    context += f"KAYNAK: {r.get('title')}\nURL: {r.get('url')}\nÖZET: {r.get('content', '')[:800]}\n\n"
                    if r.get("url"):
                        sources.append(r["url"])
                for img in res.get("images", []):
                    if _is_quality_fashion_image(img):
                        raw_images.append(img)
            except Exception as e:
                logger.warning(f"Runway search hatası: {e}")

        unique_images = list(dict.fromkeys(raw_images))[:4]

        return {
            "context": context,
            "runway_images": unique_images,
            "sources": sources,
        }

    # ─── 3. Ürün Görseli Arama ────────────────────────────────────────────

    def visual_search(self, query: str) -> dict:
        """
        Ürün görseli arama — Tavily image search.

        Returns:
            {"image_url": str | None, "page_url": str | None, "candidates": [str]}
        """
        client = _get_tavily()
        if not client:
            return {"image_url": None, "page_url": None, "candidates": []}

        search_query = f"{query} satın al abiye elbise online satış fiyatları -food -recipe"

        try:
            res = client.search(
                query=search_query,
                search_depth="advanced",
                include_images=True,
                max_results=8,
            )

            candidates = [
                img for img in res.get("images", [])
                if _is_quality_fashion_image(img)
            ]

            page_url = None
            if res.get("results"):
                page_url = res["results"][0].get("url")

            return {
                "image_url": candidates[0] if candidates else None,
                "page_url": page_url,
                "candidates": candidates[:5],
            }
        except Exception as e:
            logger.error(f"Visual search hatası: {e}")
            return {"image_url": None, "page_url": None, "candidates": []}

    # ─── 4. Bağlam Arama (Kısa) ──────────────────────────────────────────

    def context_search(self, query: str, max_results: int = 3) -> dict:
        """
        Kısa bağlam arama — intent.py yerine kullanılır.

        Returns:
            {"context": str, "sources": [str]}
        """
        client = _get_tavily()
        if not client:
            return {"context": "", "sources": []}

        try:
            res = client.search(
                query=query,
                search_depth="basic",
                include_images=False,
                max_results=max_results,
            )

            context = ""
            sources = []
            for r in res.get("results", []):
                context += f"{r.get('content', '')}\n\n"
                if r.get("url"):
                    sources.append(r["url"])

            return {"context": context.strip(), "sources": sources}
        except Exception as e:
            logger.warning(f"Context search hatası: {e}")
            return {"context": "", "sources": []}

    # ─── 5. Kapsamlı Araştırma (Paralel) ─────────────────────────────────

    async def comprehensive_research(
        self,
        topic: str,
        category: str = None,
    ) -> dict:
        """
        Tüm araştırma kaynaklarını paralel çalıştırır.

        Returns:
            {
                "market":  {market_research sonucu},
                "runway":  {runway_analysis sonucu},
                "trend_data": str,
                "category_heat": float | None,
                "combined_context": str,  # GPT-4o'ya gönderilmeye hazır
            }
        """
        loop = asyncio.get_event_loop()

        market_f = loop.run_in_executor(None, self.market_research, topic, category)
        runway_f = loop.run_in_executor(None, self.runway_analysis, topic)

        market_res, runway_res = await asyncio.gather(market_f, runway_f)

        # Birleşik context — backend'in GPT-4o'ya göndereceği veri
        combined = (
            f"{runway_res.get('context', '')}\n"
            f"===\n"
            f"{market_res.get('context', '')}"
        )

        # Kategori bilgisi
        heat = market_res.get("category_heat")
        if heat is not None:
            combined += f"\n\n=== KATEGORİ DURUMU ===\nHeat: {heat:.2f}\n"

        return {
            "market": market_res,
            "runway": runway_res,
            "trend_data": market_res.get("trend_data", ""),
            "category_heat": heat,
            "combined_context": combined,
        }

    # ─── Status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Tavily servis durumu."""
        client = _get_tavily()
        return {
            "tavily_available": client is not None,
            "search_depth": self._search_depth,
            "max_results": self._max_results,
        }


# ─── Global singleton ────────────────────────────────────────────────────────
tavily_service = TavilyResearchService()
