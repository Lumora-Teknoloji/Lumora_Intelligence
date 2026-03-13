# services/intelligence_service.py
"""
Lumora Intelligence — Engine Singleton Servisi.

PredictionEngine'i uygulama ömrü boyunca tek instance olarak tutar.
DB'den veri okur, tahmin yapar, feedback işler.
"""
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from engine.predictor import PredictionEngine
from db.reader import get_daily_metrics, get_products, get_data_summary, get_categories
from db.writer import save_predictions, save_alert, get_alerts as db_get_alerts

import config

logger = logging.getLogger(__name__)


class IntelligenceService:
    """
    Tek instance (singleton) engine wrapper.
    startup_train() → uygulama başlarken eğitim
    predict()       → kategori bazlı tahmin
    analyze()       → tekil ürün analizi
    submit_feedback() → feedback loop
    nightly_batch() → APScheduler nightly job
    """

    def __init__(self):
        self._engine: Optional[PredictionEngine] = None
        self._trained_at: Optional[datetime] = None
        self._train_data_rows: int = 0

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """DB reader 'recorded_at' döndürür, engine 'date' bekler → rename."""
        if "recorded_at" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"recorded_at": "date"})
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def startup_train(self):
        """
        Uygulama başında DB'den veri çekip engine'i eğitir.
        Veri yoksa veya yetersizse mock-free modda başlar (tahminler boş olur).
        """
        logger.info("🧠 Intelligence startup — DB'den veri çekiliyor...")
        try:
            df = get_daily_metrics(days=90)
            if df.empty:
                logger.warning("⚠ DB'de henüz veri yok — Engine mock-free modda başlıyor")
                self._engine = PredictionEngine(use_prophet=False, use_clip=False)
                return

            df = self._normalize_df(df)
            self._engine = PredictionEngine(use_prophet=False, use_clip=False)
            self._engine.train(df, verbose=False)
            self._trained_at = datetime.utcnow()
            self._train_data_rows = len(df)
            logger.info(
                f"✅ Engine eğitildi — {len(df)} satır, "
                f"{df['product_id'].nunique()} ürün, "
                f"{df['category'].nunique() if 'category' in df.columns else '?'} kategori"
            )
        except Exception as e:
            logger.error(f"❌ Engine eğitim hatası: {e}", exc_info=True)
            self._engine = PredictionEngine(use_prophet=False, use_clip=False)

    def _ensure_engine(self) -> PredictionEngine:
        if self._engine is None:
            logger.warning("Engine henüz başlatılmadı, varsayılan oluşturuluyor")
            self._engine = PredictionEngine(use_prophet=False, use_clip=False)
        return self._engine

    # ──────────────────────────────────────────────────────────────────────────
    # Tahmin API
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, category: Optional[str] = None, top_n: int = 20) -> list[dict]:
        """
        Kategori için en iyi N tahmin döndürür.
        Sonuçları DB'ye de kaydeder (intelligence_results).
        """
        engine = self._ensure_engine()

        try:
            predictions_df: pd.DataFrame = engine.predict()

            if predictions_df.empty:
                logger.warning("Engine tahmin döndürmedi")
                return []

            # Kategori filtresi
            if category and "category" in predictions_df.columns:
                predictions_df = predictions_df[predictions_df["category"] == category]

            # Top N
            if "trend_score" in predictions_df.columns:
                predictions_df = predictions_df.nlargest(top_n, "trend_score")
            else:
                predictions_df = predictions_df.head(top_n)

            # Sonuçları sözlüğe çevir
            results = []
            for _, row in predictions_df.iterrows():
                results.append({
                    "product_id":   int(row.get("product_id", 0)),
                    "category":     str(row.get("category", "")),
                    "trend_label":  str(row.get("trend_label", "")),
                    "trend_score":  float(row.get("trend_score", 0)),
                    "confidence":   float(row.get("confidence", 0)),
                    "ensemble_demand": float(row.get("ensemble_demand", 0)),
                })

            # DB'ye yaz (background — hatalar sessizce loglanır)
            try:
                save_predictions(results)
            except Exception as e:
                logger.warning(f"Tahmin DB'ye yazılamadı: {e}")

            return results

        except Exception as e:
            logger.error(f"predict() hatası: {e}", exc_info=True)
            return []

    def analyze(self, product_id: int) -> dict:
        """
        Tekil ürün analizi. Ürünün tüm metriklerini ve sinyal detaylarını döndürür.
        """
        engine = self._ensure_engine()

        try:
            # Ürüne ait son 90 günlük veri
            df = get_daily_metrics(days=90, product_ids=[product_id])
            df = self._normalize_df(df)

            if df.empty:
                return {"error": f"product_id={product_id} için veri bulunamadı"}

            # Engine'i bu ürün üzerinde çalıştır
            preds = engine.predict()

            if not preds.empty and "product_id" in preds.columns:
                prd_row = preds[preds["product_id"] == product_id]
                if not prd_row.empty:
                    row = prd_row.iloc[0]
                    return {
                        "product_id":  product_id,
                        "trend_label": str(row.get("trend_label", "UNKNOWN")),
                        "trend_score": float(row.get("trend_score", 0)),
                        "confidence":  float(row.get("confidence", 0)),
                        "signals": {
                            "ensemble_demand": float(row.get("ensemble_demand", 0)),
                            "category":        str(row.get("category", "")),
                        },
                        "data_points": len(df),
                    }

            return {
                "product_id":  product_id,
                "trend_label": "UNKNOWN",
                "trend_score": None,
                "confidence":  None,
                "signals":     {},
                "data_points": len(df),
                "note":        "Ürün tahmin listesinde yok (yetersiz veri olabilir)",
            }

        except Exception as e:
            logger.error(f"analyze() hatası: {e}", exc_info=True)
            return {"error": str(e)}

    def submit_feedback(
        self,
        product_id: int,
        sold_quantity: int,
        predicted_quantity: int,
    ) -> dict:
        """
        Gerçek satış verisiyle feedback loop'u günceller.
        """
        engine = self._ensure_engine()

        try:
            # Ürünün kategorisini bul
            df = get_daily_metrics(days=30, product_ids=[product_id])
            df = self._normalize_df(df)
            if df.empty:
                return {"status": "error", "message": "Ürün verisi bulunamadı"}

            category = df["category"].iloc[0] if "category" in df.columns else "unknown"

            engine.feedback(category, sold_quantity, predicted_quantity)

            # Büyük hata → alert üret
            error_pct = abs(sold_quantity - predicted_quantity) / max(predicted_quantity, 1) * 100
            penalty_applied = error_pct > 50

            if penalty_applied:
                save_alert({
                    "type":       "feedback_penalty",
                    "product_id": product_id,
                    "category":   category,
                    "message":    (
                        f"Büyük tahmin hatası: tahmin={predicted_quantity}, "
                        f"gerçek={sold_quantity} (%{error_pct:.0f} sapma)"
                    ),
                    "extra_data": {
                        "sold": sold_quantity,
                        "predicted": predicted_quantity,
                        "error_pct": round(error_pct, 1),
                    },
                })

            logger.info(
                f"Feedback işlendi — product={product_id}, category={category}, "
                f"sold={sold_quantity}, predicted={predicted_quantity}, "
                f"penalty={'evet' if penalty_applied else 'hayır'}"
            )

            return {
                "status":          "ok",
                "category":        category,
                "penalty_applied": penalty_applied,
                "error_pct":       round(error_pct, 1),
            }

        except Exception as e:
            logger.error(f"submit_feedback() hatası: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # Scheduler Jobs
    # ──────────────────────────────────────────────────────────────────────────

    async def nightly_batch(self):
        """
        APScheduler tarafından her gece 02:00'de çalıştırılır.
        Sıra:
          1. Rank momentum güncelle  (daily_metrics.rank_change_*)
          2. Tüm kategorileri score'la
          3. Kategori heat map kaydet  (category_daily_signals)
          4. Alertleri üret
        """
        from db.writer import update_rank_momentum, save_category_signal

        logger.info("🌙 Nightly batch başladı")
        try:
            # ── 1. Rank momentum ─────────────────────────────────────────────
            momentum_updated = update_rank_momentum()
            logger.info(f"  Rank momentum: {momentum_updated} ürün güncellendi")

            # ── 2. Kategori scoring ──────────────────────────────────────────
            categories = get_categories()
            if not categories:
                logger.warning("Aktif kategori yok, batch atlandı")
                return

            all_results = []
            for cat in categories:
                preds = self.predict(category=cat, top_n=500)
                all_results.extend(preds)

                if not preds:
                    continue

                # ── 3. Kategori heat map ─────────────────────────────────────
                rising  = sum(1 for p in preds if p.get("trend_label") in ("TREND", "POTANSIYEL"))
                falling = sum(1 for p in preds if p.get("trend_label") == "DUSEN")
                total   = len(preds)

                # Basit heat = (rising - falling) / total  → normalize tanh
                import math
                raw_heat  = (rising - falling) / max(total, 1)
                cat_heat  = math.tanh(raw_heat * 3)   # [-1, +1]

                save_category_signal({
                    "search_term":     cat,
                    "total_products":  total,
                    "rising_count":    rising,
                    "falling_count":   falling,
                    "new_entrants":    sum(1 for p in preds if p.get("is_new_entrant")),
                    "avg_fav_change":  0.0,   # DailyMetric'ten gelecek (Faz 2)
                    "avg_rank_change": 0.0,   # momentum hesaplanınca eklenecek
                    "category_heat":   round(cat_heat, 3),
                })

                # ── 4. Rank spike alertleri ──────────────────────────────────
                for p in preds:
                    if p.get("trend_score", 0) > 90:
                        save_alert({
                            "type":       "rank_spike",
                            "product_id": p["product_id"],
                            "category":   cat,
                            "message":    f"Yüksek trend skoru: {p['trend_score']:.1f}",
                            "extra_data": p,
                        })

            logger.info(
                f"✅ Nightly batch tamamlandı — "
                f"{len(all_results)} ürün, {len(categories)} kategori"
            )

            # ── Backend'e bildir (fire & forget) ─────────────────────────
            # Frontend'in trend listesini yenileyebilmesi için callback gönder
            await self._notify_backend_callback(
                event="scoring_complete",
                trend_count=sum(1 for r in all_results if r.get("trend_label") == "TREND"),
                category=None,
            )

        except Exception as e:
            logger.error(f"Nightly batch hatası: {e}", exc_info=True)

    async def _notify_backend_callback(
        self,
        event: str = "scoring_complete",
        category: str = None,
        trend_count: int = 0,
    ):
        """Backend /api/intelligence/callback endpoint'ine bildirim gönderir."""
        import asyncio, urllib.request, json as _json
        from datetime import datetime, timezone

        def _post():
            try:
                from config import BACKEND_CALLBACK_URL
                body = _json.dumps({
                    "event":       event,
                    "category":    category,
                    "trend_count": trend_count,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                }).encode()
                req = urllib.request.Request(
                    BACKEND_CALLBACK_URL,
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5).read()
                logger.debug("Backend callback bildirimi gönderildi")
            except Exception as e:
                logger.debug(f"Backend callback gönderilemedi (normal): {e}")

        # async context içinde threading kullan — blocking IO olmasın
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _post)


    async def weekly_retrain(self):
        """
        APScheduler tarafından her Pazar 03:00'de çalıştırılır.
        30+ günlük veri varsa CatBoost'u yeniden eğitir.
        """
        logger.info("📅 Haftalık retraining başladı")
        try:
            summary = get_data_summary()
            data_days = summary.get("data_days", 0)

            if data_days < config.MIN_DAYS_FOR_CATBOOST:
                logger.info(
                    f"⏳ Retraining atlandı — sadece {data_days} günlük veri var "
                    f"(min {config.MIN_DAYS_FOR_CATBOOST} gün gerekli)"
                )
                return

            df = get_daily_metrics(days=data_days)
            df = self._normalize_df(df)
            if df.empty:
                return

            engine = self._ensure_engine()
            engine.train(df, verbose=False)
            self._trained_at = datetime.utcnow()
            self._train_data_rows = len(df)

            logger.info(f"✅ Retraining tamamlandı — {len(df)} satır, {data_days} günlük veri")

        except Exception as e:
            logger.error(f"Weekly retraining hatası: {e}", exc_info=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Status / Health
    # ──────────────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Engine durumu (health endpoint için)."""
        if self._engine is None:
            return {"engine_trained": False}

        engine_status = self._engine.status()
        db_summary = get_data_summary()

        return {
            "engine_trained":    engine_status.get("catboost_trained", False),
            "trained_at":        self._trained_at.isoformat() if self._trained_at else None,
            "train_data_rows":   self._train_data_rows,
            "prophet_enabled":   engine_status.get("prophet_enabled", False),
            "clip_enabled":      engine_status.get("clip_enabled", False),
            "kalman_categories": len(engine_status.get("kalman_category_states", {})),
            "db_summary":        db_summary,
        }

    def get_alerts(self, unread_only: bool = False) -> list[dict]:
        return db_get_alerts(unread_only=unread_only)


# ─── Global singleton ─────────────────────────────────────────────────────────
intelligence_service = IntelligenceService()
