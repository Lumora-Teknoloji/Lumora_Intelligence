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
from engine.category_classifier import CategoryAutoClassifier
from db.reader import get_daily_metrics, get_products, get_data_summary, get_categories, get_category_stats
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
        Uygulama başında DB'den veri çekip engine'i hazırlar.
        1. Kategorileri otomatik sınıflandır → categories_registry'ye yaz
        2. DB profillerini yükle
        3. Mevcut modelleri yükle (varsa) — yeniden eğitme!
        4. Sadece lightweight init: Z-Score, Feature Eng, Kalman
        
        NOT: Tam retraining weekly_retrain() ile yapılır (Pazar 03:00).
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

            # ── Kategori Otomatik Sınıflandırma ──────────────────────────────
            try:
                stats = get_category_stats()
                if stats:
                    classifier = CategoryAutoClassifier()
                    classify_results = classifier.classify_all(stats)
                    logger.info(
                        f"🏷️ {len(classify_results)} kategori sınıflandırıldı → "
                        f"categories_registry güncellendi"
                    )
            except Exception as e:
                logger.warning(f"Kategori sınıflandırma hatası (devam): {e}")

            # ── DB profillerini registry'ye yükle ────────────────────────────
            self._engine.registry.load_db_profiles()

            # ── Daha önce eğitilmiş per-category modelleri yükle ─────────────
            self._engine.registry.load_all()
            loaded = self._engine.registry.trained_categories
            
            if loaded:
                logger.info(f"📦 {len(loaded)} mevcut model yüklendi: {loaded}")
                
                # ── Lightweight init: sadece Z-Score + Feature + Kalman ───────
                # CatBoost modelleri zaten diskten yüklendi, tekrar eğitmeye gerek yok
                logger.info("⚡ Hızlı başlatma — mevcut modeller kullanılıyor (retraining atlandı)")
                
                try:
                    # Kolon adı normalizasyonu (train() içindeki ile aynı mantık)
                    df_work = df.copy()
                    col_renames = {}
                    if "product_category" in df_work.columns and "category" not in df_work.columns:
                        col_renames["product_category"] = "category"
                    elif "product_category" in df_work.columns and "category" in df_work.columns and df_work["category"].isna().all():
                        df_work.drop(columns=["category"], inplace=True)
                        col_renames["product_category"] = "category"
                    if "recorded_at" in df_work.columns and "date" not in df_work.columns:
                        col_renames["recorded_at"] = "date"
                    if col_renames:
                        df_work = df_work.rename(columns=col_renames)
                    if "date" in df_work.columns:
                        df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
                    
                    # Z-Score + Feature Engineering (predict için gerekli)
                    df_clean = self._engine.zscore.filter_errors(df_work)
                    df_clean = self._engine.zscore.detect_anomalies(df_clean)
                    df_feat = self._engine.feature_eng.build_features(df_clean)
                    
                    # Kalman filtreler (kategori + ürün bazlı)
                    for category in df_feat["category"].unique():
                        cat_data = df_feat[df_feat["category"] == category]
                        daily_eng = cat_data.groupby("date")["engagement_score"].mean().sort_index().values
                        self._engine.kalman.process_series(category, daily_eng.tolist())
                    
                    product_count = 0
                    for pid in df_feat["product_id"].unique():
                        p_data = df_feat[df_feat["product_id"] == pid].sort_values("date")
                        if len(p_data) < 3:
                            continue
                        cart_series = p_data["cart_count"].values.tolist()
                        self._engine.kalman.process_product(pid, cart_series)
                        product_count += 1
                    
                    self._engine.is_trained = True
                    self._engine._last_training_data = df_feat
                    self._trained_at = datetime.utcnow()
                    self._train_data_rows = len(df)
                    
                    logger.info(
                        f"✅ Engine hazır (hızlı mod) — {len(df)} satır, "
                        f"{df['product_id'].nunique()} ürün, "
                        f"{len(loaded)} kategori modeli, "
                        f"{product_count} ürün Kalman"
                    )
                except Exception as init_err:
                    logger.error(
                        f"❌ Lightweight init hatası: {init_err}", exc_info=True
                    )
                    # Fallback: modeller yüklü, sadece is_trained set et
                    # predict() en azından registry modelleri ile çalışabilir
                    logger.info("🔄 Fallback: tam eğitim yapılıyor...")
                    self._engine.train(df, verbose=False)
                    self._trained_at = datetime.utcnow()
                    self._train_data_rows = len(df)
            else:
                # Hiç model yoksa → tam eğitim yap (ilk sefer)
                logger.info("🔄 İlk çalıştırma — tam eğitim yapılıyor...")
                self._engine.train(df, verbose=False)
                self._trained_at = datetime.utcnow()
                self._train_data_rows = len(df)
                logger.info(
                    f"✅ Engine eğitildi — {len(df)} satır, "
                    f"{df['product_id'].nunique()} ürün, "
                    f"{df['category'].nunique() if 'category' in df.columns else '?'} kategori"
                )

            # Per-category model durumu
            status = self._engine.registry.status()
            logger.info(
                f"🏠 Model Registry: "
                f"{status['trained_categories']}/{status['total_categories']} kategori eğitildi"
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
        Zenginleştirilmiş ürün detayları ile birlikte döndürür.
        """
        engine = self._ensure_engine()

        try:
            predictions_df: pd.DataFrame = engine.predict()

            if predictions_df.empty:
                logger.warning("Engine tahmin döndürmedi")
                return []

            # Kategori filtresi (Substring match)
            if category and "category" in predictions_df.columns:
                predictions_df = predictions_df[predictions_df["category"].str.contains(category, case=False, na=False)]

            # Top N
            if "trend_score" in predictions_df.columns:
                predictions_df = predictions_df.nlargest(top_n, "trend_score")
            else:
                predictions_df = predictions_df.head(top_n)

            # Ürün ID'lerini topla
            product_ids = predictions_df["product_id"].unique().tolist()

            product_details = {}
            latest_metrics = {}
            new_entrants_map = {}

            if product_ids:
                try:
                    from sqlalchemy import text as sql_text
                    from db.connection import engine as db_engine

                    with db_engine.connect() as conn:
                        combined_sql = sql_text("""
                            WITH latest_metrics AS (
                                SELECT DISTINCT ON (product_id)
                                    product_id, price, discounted_price, discount_rate,
                                    cart_count, favorite_count, view_count,
                                    rating_count, avg_rating, search_rank,
                                    engagement_score, popularity_score, sales_velocity,
                                    rank_change_1d, rank_change_3d, rank_velocity,
                                    momentum_score
                                FROM daily_metrics
                                WHERE product_id = ANY(:pids)
                                ORDER BY product_id, recorded_at DESC
                            ),
                            latest_ne AS (
                                SELECT DISTINCT ON (product_id) product_id, is_new_entrant
                                FROM daily_metrics
                                WHERE product_id = ANY(:pids) AND is_new_entrant IS NOT NULL
                                ORDER BY product_id, recorded_at DESC
                            )
                            SELECT 
                                p.id, p.product_code, p.name, p.brand, p.seller, p.category,
                                p.category_tag, p.url, p.image_url,
                                p.last_price, p.last_discount_rate, p.last_engagement_score,
                                p.avg_sales_velocity, p.dominant_color, p.fabric_type, p.fit_type,
                                p.review_summary, p.sizes, p.attributes,
                                dm.price, dm.discounted_price, dm.discount_rate,
                                dm.cart_count, dm.favorite_count, dm.view_count,
                                dm.rating_count, dm.avg_rating, dm.search_rank,
                                dm.engagement_score AS m_engagement_score, 
                                dm.popularity_score, 
                                dm.sales_velocity AS m_sales_velocity,
                                dm.rank_change_1d, dm.rank_change_3d, dm.rank_velocity,
                                dm.momentum_score, ne.is_new_entrant
                            FROM products p
                            LEFT JOIN latest_metrics dm ON p.id = dm.product_id
                            LEFT JOIN latest_ne ne ON p.id = ne.product_id
                            WHERE p.id = ANY(:pids)
                        """)
                        rows = conn.execute(combined_sql, {"pids": product_ids}).fetchall()

                        for row in rows:
                            pid = row[0]
                            # Products 
                            product_details[pid] = {
                                "product_code": row[1],
                                "name": row[2],
                                "brand": row[3],
                                "seller": row[4],
                                "category": row[5],
                                "category_tag": row[6],
                                "url": row[7],
                                "image_url": row[8],
                                "last_price": float(row[9]) if row[9] is not None else None,
                                "last_discount_rate": float(row[10]) if row[10] is not None else None,
                                "engagement_score": float(row[11]) if row[11] is not None else None,
                                "avg_sales_velocity": float(row[12]) if row[12] is not None else None,
                                "dominant_color": row[13],
                                "fabric_type": row[14],
                                "fit_type": row[15],
                                "review_summary": row[16],
                                "sizes": row[17],
                                "attributes": row[18],
                            }
                            # Metrics
                            latest_metrics[pid] = {
                                "price": float(row[19]) if row[19] is not None else None,
                                "discounted_price": float(row[20]) if row[20] is not None else None,
                                "discount_rate": float(row[21]) if row[21] is not None else None,
                                "cart_count": int(row[22]) if row[22] is not None else 0,
                                "favorite_count": int(row[23]) if row[23] is not None else 0,
                                "view_count": int(row[24]) if row[24] is not None else 0,
                                "rating_count": int(row[25]) if row[25] is not None else 0,
                                "avg_rating": float(row[26]) if row[26] is not None else None,
                                "search_rank": int(row[27]) if row[27] is not None else None,
                                "engagement_score": float(row[28]) if row[28] is not None else None,
                                "popularity_score": float(row[29]) if row[29] is not None else None,
                                "sales_velocity": float(row[30]) if row[30] is not None else None,
                                "rank_change_1d": int(row[31]) if row[31] is not None else None,
                                "rank_change_3d": int(row[32]) if row[32] is not None else None,
                                "rank_velocity": float(row[33]) if row[33] is not None else None,
                                "momentum_score": float(row[34]) if row[34] is not None else None,
                            }
                            # New Entrant
                            if row[35] is not None:
                                new_entrants_map[pid] = bool(row[35])
                except Exception as e:
                    logger.warning(f"Ürün verileri veritabanından çekilemedi: {e}")

            # Sonuçları sözlüğe çevir — ZENGİNLEŞTİRİLMİŞ
            results = []
            for _, row in predictions_df.iterrows():
                pid = int(row.get("product_id", 0))
                detail = product_details.get(pid, {})
                metrics = latest_metrics.get(pid, {})

                # JSONB attributes'tan ek detaylar çıkar
                attrs = detail.get("attributes") or {}
                if isinstance(attrs, str):
                    import json as _j
                    try: attrs = _j.loads(attrs)
                    except: attrs = {}

                result = {
                    # Tahmin verileri
                    "product_id":       pid,
                    "trend_label":      str(row.get("trend_label", "")),
                    "trend_score":      float(row.get("trend_score", 0)),
                    "confidence":       float(row.get("confidence", 0)),
                    "ensemble_demand":  float(row.get("ensemble_demand", 0)),

                    # Ürün kimlik bilgileri
                    "product_code":     detail.get("product_code"),
                    "name":             detail.get("name"),
                    "brand":            detail.get("brand"),
                    "seller":           detail.get("seller"),
                    "category":         detail.get("category") or str(row.get("category", "")),
                    "category_tag":     detail.get("category_tag"),
                    "url":              detail.get("url"),
                    "image_url":        detail.get("image_url"),

                    # Fiyat bilgileri
                    "price":            metrics.get("price") or detail.get("last_price"),
                    "discounted_price": metrics.get("discounted_price"),
                    "discount_rate":    metrics.get("discount_rate") or detail.get("last_discount_rate"),

                    # Stil özellikleri
                    "dominant_color":   detail.get("dominant_color"),
                    "fabric_type":      detail.get("fabric_type"),
                    "fit_type":         detail.get("fit_type"),
                    "sizes":            detail.get("sizes"),

                    # JSONB ek özellikler — zengin detay
                    "attributes":       attrs,
                    "review_summary":   detail.get("review_summary"),

                    # Performans metrikleri
                    "favorite_count":   metrics.get("favorite_count", 0),
                    "cart_count":       metrics.get("cart_count", 0),
                    "view_count":       metrics.get("view_count", 0),
                    "rating_count":     metrics.get("rating_count", 0),
                    "avg_rating":       metrics.get("avg_rating"),
                    "search_rank":      metrics.get("search_rank"),
                    "engagement_score": metrics.get("engagement_score") or detail.get("engagement_score"),
                    "popularity_score": metrics.get("popularity_score"),
                    "sales_velocity":   metrics.get("sales_velocity") or detail.get("avg_sales_velocity"),

                    # Rank momentum verileri
                    "rank_change_1d":   metrics.get("rank_change_1d"),
                    "rank_change_3d":   metrics.get("rank_change_3d"),
                    "rank_velocity":    metrics.get("rank_velocity"),
                    "momentum_score":   metrics.get("momentum_score"),

                    # Intelligence sinyalleri
                    "is_new_entrant":   new_entrants_map.get(pid, False),
                }
                results.append(result)

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

            # Kolon normalizasyonu (Feature engine için)
            if "product_category" in df.columns and "category" not in df.columns:
                df = df.rename(columns={"product_category": "category"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # Veriyi temizle ve sadece bu ürün için feature'ları çıkar
            df_clean = engine.zscore.filter_errors(df)
            df_clean = engine.zscore.detect_anomalies(df_clean)
            
            if df_clean.empty:
                return {"error": f"product_id={product_id} analiz edilebilecek yeterli temiz veriye sahip değil"}
                
            df_feat = engine.feature_eng.build_features(df_clean)

            # Sadece bu ürün üzerinde çalıştır (Global 10k product loop'undan kurtarır)
            preds = engine.predict(df=df_feat)

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

            engine.feedback(
                category=category, 
                actual_sales=sold_quantity, 
                predicted_demand=predicted_quantity, 
                product_id=product_id
            )

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

    def _calc_avg_fav_change(self, category: str) -> float:
        """Kategorideki son 1 günlük ortalama favori değişimini hesaplar."""
        from sqlalchemy import text as sql_text
        from db.connection import engine as db_engine
        try:
            sql = sql_text("""
                SELECT ROUND(AVG(t.fav_diff)::numeric, 2)
                FROM (
                    SELECT
                        dm.product_id,
                        dm.favorite_count - LAG(dm.favorite_count)
                            OVER (PARTITION BY dm.product_id ORDER BY dm.recorded_at) AS fav_diff
                    FROM daily_metrics dm
                    WHERE dm.search_term = :cat
                      AND dm.recorded_at >= CURRENT_DATE - INTERVAL '2 days'
                      AND dm.favorite_count IS NOT NULL
                ) t
                WHERE t.fav_diff IS NOT NULL
            """)
            with db_engine.connect() as conn:
                val = conn.execute(sql, {"cat": category}).scalar()
                return float(val) if val else 0.0
        except Exception:
            return 0.0

    def _calc_avg_rank_change(self, category: str) -> float:
        """Kategorideki son 1 günlük ortalama rank değişimini hesaplar."""
        from sqlalchemy import text as sql_text
        from db.connection import engine as db_engine
        try:
            sql = sql_text("""
                SELECT ROUND(AVG(rank_change_1d)::numeric, 2)
                FROM daily_metrics
                WHERE search_term = :cat
                  AND recorded_at >= CURRENT_DATE
                  AND rank_change_1d IS NOT NULL
            """)
            with db_engine.connect() as conn:
                val = conn.execute(sql, {"cat": category}).scalar()
                return float(val) if val else 0.0
        except Exception:
            return 0.0

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
            # ── 0. Kategori yeniden sınıflandırma ────────────────────────────
            try:
                stats = get_category_stats()
                if stats:
                    classifier = CategoryAutoClassifier()
                    classifier.classify_all(stats)
                    # Engine registry'yi güncelle
                    engine = self._ensure_engine()
                    engine.registry._profiles.clear()  # Cache temizle
                    engine.registry.load_db_profiles()
            except Exception as e:
                logger.warning(f"Nightly classify hatası (devam): {e}")

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
                    "avg_fav_change":  self._calc_avg_fav_change(cat),
                    "avg_rank_change": self._calc_avg_rank_change(cat),
                    "category_heat":   round(cat_heat, 3),
                })

                # Kategori registry güncelle (yaşam döngüsü)
                try:
                    from db.writer import upsert_category_registry
                    engine = self._ensure_engine()
                    profile = engine.registry.get_profile(cat)
                    kalman_state = engine.kalman.get_state(cat)

                    # Günlük veri sayısını hesapla
                    data_summary = get_data_summary()
                    data_days = data_summary.get("data_days", 0)

                    # Yaşam döngüsü
                    if data_days >= 365:
                        lifecycle = "MATURE"
                    elif data_days >= 90:
                        lifecycle = "HOT"
                    elif data_days >= 30:
                        lifecycle = "WARMING"
                    else:
                        lifecycle = "COLD"

                    upsert_category_registry({
                        "search_term":    cat,
                        "profile_type":   profile.get("profile_type", "mid_fashion"),
                        "lifecycle":      lifecycle,
                        "data_days":      data_days,
                        "total_products": total,
                        "kalman_state":   kalman_state,
                    })
                except Exception:
                    pass  # Opsiyonel — sessizce geç

                # ── 4. Alertler ───────────────────────────────────────────────
                for p in preds:
                    score = p.get("trend_score", 0)
                    pid = p["product_id"]

                    # 4a. Rank spike (mevcut)
                    if score > 90:
                        save_alert({
                            "type":       "rank_spike",
                            "product_id": pid,
                            "category":   cat,
                            "message":    f"Yüksek trend skoru: {score:.1f}",
                            "extra_data": p,
                        })

                    # 4b. Viral start — yeni giren + yüksek skor
                    if p.get("is_new_entrant") and score > 80:
                        save_alert({
                            "type":       "viral_start",
                            "product_id": pid,
                            "category":   cat,
                            "message":    f"Yeni giriş + yüksek skor: {score:.1f}",
                            "extra_data": p,
                        })

                # 4c. Rank drop — herhangi birinde 3 günlük ciddi kötüleşme
                try:
                    from sqlalchemy import text as _t
                    from db.connection import engine as _dbe
                    with _dbe.connect() as _conn:
                        drop_sql = _t("""
                            SELECT DISTINCT ON (product_id) product_id, rank_change_3d
                            FROM daily_metrics
                            WHERE search_term = :cat
                              AND recorded_at >= CURRENT_DATE
                              AND rank_change_3d > 300
                            ORDER BY product_id, recorded_at DESC
                            LIMIT 10
                        """)
                        drops = _conn.execute(drop_sql, {"cat": cat}).fetchall()
                        for drop_row in drops:
                            save_alert({
                                "type":       "rank_drop",
                                "product_id": drop_row[0],
                                "category":   cat,
                                "message":    f"3 günde {int(drop_row[1])} sıra kötüleşme",
                                "extra_data": {"rank_change_3d": float(drop_row[1])},
                            })
                except Exception:
                    pass  # Opsiyonel — sessizce geç

                # 4d. Category heat alert
                if cat_heat > 0.8:
                    save_alert({
                        "type":       "category_heat",
                        "product_id": None,
                        "category":   cat,
                        "message":    f"Kategori ısınıyor: heat={cat_heat:.2f}",
                        "extra_data": {"heat": cat_heat, "rising": rising, "total": total},
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
        import httpx
        from datetime import datetime, timezone

        try:
            from config import BACKEND_CALLBACK_URL
            payload = {
                "event":       event,
                "category":    category,
                "trend_count": trend_count,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(BACKEND_CALLBACK_URL, json=payload)
            logger.debug("Backend callback bildirimi gönderildi")
        except Exception as e:
            logger.debug(f"Backend callback gönderilemedi (normal): {e}")


    async def weekly_retrain(self):
        """
        APScheduler tarafından her Pazar 03:00'de çalıştırılır.
        30+ günlük veri varsa CatBoost'u per-category yeniden eğitir.
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

            # Per-category retraining
            cat_results = engine.registry.train_all(df, verbose=False)
            trained = sum(1 for v in cat_results.values() if v)

            # Kalman'ları da güncelle
            for category in df["category"].unique():
                cat_data = df[df["category"] == category]
                daily_eng = cat_data.groupby("date")["engagement_score"].mean().sort_index().values
                engine.kalman.process_series(category, daily_eng.tolist())

            self._trained_at = datetime.utcnow()
            self._train_data_rows = len(df)

            logger.info(
                f"✅ Retraining tamamlandı — {trained}/{len(cat_results)} kategori, "
                f"{len(df)} satır, {data_days} günlük veri"
            )

            # Modelleri kaydet
            engine.registry.save_all()

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

        # Per-category model bilgisi
        registry_info = engine_status.get("registry_status", {})

        # Feedback istatistikleri
        feedback_stats = {}
        try:
            from db.writer import get_feedback_stats
            feedback_stats = get_feedback_stats()
        except Exception:
            pass

        return {
            "engine_trained":       engine_status.get("catboost_trained", False),
            "trained_at":           self._trained_at.isoformat() if self._trained_at else None,
            "train_data_rows":      self._train_data_rows,
            "prophet_enabled":      engine_status.get("prophet_enabled", False),
            "clip_enabled":         engine_status.get("clip_enabled", False),
            "kalman_categories":    len(engine_status.get("kalman_category_states", {})),
            "db_summary":           db_summary,
            # Yeni alanlar
            "per_category_models":  registry_info,
            "feedback_stats":       feedback_stats,
        }

    def get_alerts(self, unread_only: bool = False) -> list[dict]:
        return db_get_alerts(unread_only=unread_only)


# ─── Global singleton ─────────────────────────────────────────────────────────
intelligence_service = IntelligenceService()
