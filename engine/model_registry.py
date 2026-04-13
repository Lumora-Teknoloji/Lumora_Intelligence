# engine/model_registry.py
"""
Per-category CatBoost model yönetimi.

Her kategori kendi:
  - DemandPredictor instance'ı
  - Z-Score eşiği
  - Scoring ağırlıkları
  - Eğitim verisine sahiptir

Yeni kategori geldiğinde otomatik profil atanır ve model oluşturulur.

Kullanım:
    registry = CategoryModelRegistry()
    registry.train_category("crop", df_crop)
    predictions = registry.predict_category("crop", df_crop)
"""
import os
import logging
from typing import Optional

import pandas as pd

from algorithms.catboost_model import DemandPredictor
from engine.slug import slugify
from config import (
    CATEGORY_PROFILES,
    CATEGORY_TAG_MAP,
    DEFAULT_CATEGORY_PROFILE,
    CATBOOST_ITERATIONS,
    CATBOOST_DEPTH,
    CATBOOST_LEARNING_RATE,
)

logger = logging.getLogger(__name__)


class CategoryModelRegistry:
    """
    Dinamik kategori modeli yönetimi.

    Her kategori:
      - Kendi CatBoost modeline sahip
      - Kendi profil parametrelerini kullanır
      - Bağımsız olarak eğitilir ve kaydedilir

    Yaşam döngüsü:
      COLD    (0-30 gün)  → Model yok, sadece rule-based sinyal
      WARMING (30+ gün)   → CatBoost ilk eğitim
      HOT     (90+ gün)   → Tam ensemble + feedback
      MATURE  (365+ gün)  → Prophet mevsimsellik
    """

    def __init__(self, model_base_dir: str = "models"):
        self._models: dict[str, DemandPredictor] = {}
        self._profiles: dict[str, dict] = {}       # category → resolved profile
        self._db_profiles: dict[str, dict] = {}    # DB'den yüklenen profil bilgileri
        self._train_stats: dict[str, dict] = {}     # category → {rows, products, trained_at}
        self._model_base_dir = model_base_dir
        os.makedirs(model_base_dir, exist_ok=True)

    # ─── Profil Yönetimi ──────────────────────────────────────────────────────

    def load_db_profiles(self):
        """DB'deki categories_registry'den profil bilgilerini yükler."""
        try:
            from db.reader import get_category_profiles
            self._db_profiles = get_category_profiles()
            logger.info(f"📋 {len(self._db_profiles)} profil DB'den yüklendi")
        except Exception as e:
            logger.warning(f"DB profil yükleme hatası: {e}")
            self._db_profiles = {}

    def get_profile(self, category: str) -> dict:
        """
        Kategori profilini döndürür.
        Öncelik: cache → DB (profile_type + overrides merge) → tag map → default
        """
        if category in self._profiles:
            return self._profiles[category]

        db_entry = self._db_profiles.get(category, {})

        # 1. DB'den profil tipi
        if db_entry:
            profile_key = db_entry.get("profile_type", DEFAULT_CATEGORY_PROFILE)
        # 2. Tag map fallback
        elif category in CATEGORY_TAG_MAP:
            profile_key = CATEGORY_TAG_MAP[category]
        # 3. Default
        else:
            profile_key = DEFAULT_CATEGORY_PROFILE

        profile = CATEGORY_PROFILES.get(profile_key, CATEGORY_PROFILES[DEFAULT_CATEGORY_PROFILE]).copy()
        profile["profile_type"] = profile_key
        profile["group_name"] = db_entry.get("group_name")

        # DB overrides merge (sadece farklılaşan değerler)
        overrides = db_entry.get("overrides", {})
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(profile.get(key), dict):
                    profile[key] = {**profile[key], **value}  # Nested dict merge
                else:
                    profile[key] = value
            logger.debug(f"[{category}] override uygulandı: {list(overrides.keys())}")

        self._profiles[category] = profile
        return profile

    def _resolve_model_key(self, category: str) -> str:
        """
        Model dizini için slug key döndürür.
        Grup varsa → slug(grup adı) (abiye nikah → models/abiye/)
        Yoksa → slug(kategori adı) (gecelik → models/gecelik/)
        """
        profile = self.get_profile(category)
        raw_key = profile.get("group_name") or category
        return slugify(raw_key)

    def get_zscore_threshold(self, category: str) -> float:
        """Kategori için Z-Score anomali eşiğini döndürür."""
        return self.get_profile(category).get("zscore_threshold", 3.0)

    def get_score_weights(self, category: str) -> dict:
        """Kategori için scoring ağırlıklarını döndürür."""
        return self.get_profile(category).get(
            "score_weights",
            {"growth": 0.40, "velocity": 0.35, "demand": 0.25},
        )

    def get_label_thresholds(self, category: str) -> dict:
        """Kategori için etiket eşiklerini döndürür."""
        return self.get_profile(category).get(
            "label_thresholds",
            {"trend": 70, "potansiyel": 40, "stabil": 20},
        )

    # ─── Model Yönetimi ──────────────────────────────────────────────────────

    def get_or_create(self, category: str) -> DemandPredictor:
        """
        Kategorinin modelini döndürür, yoksa yeni oluşturur.
        Grubu varsa grubun modelini paylaşır: models/{group_name}/
        """
        model_key = self._resolve_model_key(category)

        if model_key in self._models:
            return self._models[model_key]

        model_dir = os.path.join(self._model_base_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)

        predictor = DemandPredictor(model_dir=model_dir)

        model_path = os.path.join(model_dir, "catboost_v1.cbm")
        if os.path.exists(model_path):
            try:
                predictor.load(version=1)
                logger.info(f"✅ Model yüklendi: {model_key} ({model_path})")
            except Exception as e:
                logger.warning(f"⚠ Model yüklenemedi ({model_key}): {e}")

        self._models[model_key] = predictor
        return predictor

    def train_category(self, category: str, df: pd.DataFrame,
                       verbose: bool = True) -> bool:
        """
        Kategoriyi kendi profil parametreleriyle eğitir.

        Args:
            category: Kategori adı (crop, tayt, ...)
            df: Bu kategoriye ait daily_metrics verisi
            verbose: Eğitim çıktılarını göster

        Returns:
            True ise eğitim başarılı
        """
        profile = self.get_profile(category)
        n_products = df["product_id"].nunique() if "product_id" in df.columns else 0

        # Sparse profil CatBoost kullanmaz
        if not profile.get("use_catboost", True):
            logger.info(
                f"⏭️ [{category}] profil={profile['profile_type']} — "
                f"CatBoost kullanılmıyor, eğitim atlandı ({n_products} ürün)"
            )
            return False

        if n_products < profile.get("min_products", 10):
            logger.info(
                f"⏳ {category}: {n_products} ürün — "
                f"min {profile['min_products']} gerekli, eğitim atlandı"
            )
            return False

        predictor = self.get_or_create(category)

        # Per-category CatBoost parametrelerini override et
        from catboost import CatBoostRegressor
        original_depth = predictor.model.get_param("depth") if predictor.model else None

        if verbose:
            logger.info(
                f"🧠 [{category}] Eğitim başlıyor — "
                f"profil={profile['profile_type']}, "
                f"depth={profile['catboost_depth']}, "
                f"iter={profile['catboost_iterations']}, "
                f"{len(df)} kayıt, {n_products} ürün"
            )

        # DemandPredictor.train() çağırılmadan önce config override
        import config as cfg
        orig_iter = cfg.CATBOOST_ITERATIONS
        orig_depth = cfg.CATBOOST_DEPTH
        try:
            cfg.CATBOOST_ITERATIONS = profile.get("catboost_iterations", orig_iter)
            cfg.CATBOOST_DEPTH = profile.get("catboost_depth", orig_depth)
            success = predictor.train(df, verbose=verbose)
        finally:
            # Global config'i geri yükle
            cfg.CATBOOST_ITERATIONS = orig_iter
            cfg.CATBOOST_DEPTH = orig_depth

        if success:
            predictor.save(version=1)
            self._train_stats[category] = {
                "rows": len(df),
                "products": n_products,
                "profile": profile["profile_type"],
            }

            # DB'ye model versiyonu kaydet
            try:
                from db.writer import save_model_version
                save_model_version({
                    "category":       category,
                    "version":        1,
                    "profile_type":   profile["profile_type"],
                    "train_rows":     len(df),
                    "train_products": n_products,
                    "r2_score":       predictor.metrics.get("r2"),
                    "mae":            predictor.metrics.get("mae"),
                    "mape":           predictor.metrics.get("mape"),
                    "feature_count":  len(predictor._feature_names),
                    "top_features":   dict(list(predictor.feature_importance.items())[:10]),
                    "model_path":     os.path.join(predictor.model_dir, "catboost_v1.cbm"),
                })
            except Exception as e:
                logger.warning(f"Model version DB kaydı başarısız ({category}): {e}")

            logger.info(
                f"✅ [{category}] Eğitim tamamlandı — "
                f"R²={predictor.metrics.get('r2', '?')}"
            )

        return success

    def predict_category(self, category: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kategorinin kendi modeli ile tahmin yapar.
        Model eğitilmemişse boş DataFrame döner.
        """
        predictor = self.get_or_create(category)
        if not predictor.is_trained:
            logger.debug(f"⏳ [{category}] Model henüz eğitilmedi — atlanıyor")
            return pd.DataFrame()

        return predictor.predict(df)

    # ─── Toplu İşlemler ───────────────────────────────────────────────────────

    def train_all(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Tüm kategorileri kendi profilleriyle eğitir.

        Args:
            df: Tüm kategorileri içeren daily_metrics verisi
                (category kolonu zorunlu)

        Returns:
            {category: bool (başarı durumu)}
        """
        if "category" not in df.columns:
            logger.error("train_all: 'category' kolonu yok")
            return {}

        categories = [c for c in df["category"].unique() if c is not None]
        results = {}

        # ── Grup bazlı birleştirme: aynı model_key → tek eğitim ────────
        from collections import defaultdict
        model_groups = defaultdict(list)  # model_key → [cat1, cat2, ...]
        for cat in categories:
            key = self._resolve_model_key(cat)
            model_groups[key].append(cat)

        trained_keys = set()

        if verbose:
            n_groups = sum(1 for cats in model_groups.values() if len(cats) > 1)
            n_standalone = sum(1 for cats in model_groups.values() if len(cats) == 1)
            logger.info(
                f"🏭 {len(categories)} kategori → "
                f"{len(model_groups)} model ({n_groups} grup + {n_standalone} bağımsız)"
            )

        for model_key, group_cats in model_groups.items():
            # Zaten eğitilmişse atla (güvenlik)
            if model_key in trained_keys:
                continue

            # Grup verilerini birleştir
            if len(group_cats) > 1:
                merged_df = pd.concat(
                    [df[df["category"] == c] for c in group_cats],
                    ignore_index=True,
                )
                representative_cat = group_cats[0]
                if verbose:
                    logger.info(
                        f"  📦 [{model_key}] {len(group_cats)} alt-kategori birleştirildi: "
                        f"{group_cats} → {len(merged_df)} kayıt"
                    )
            else:
                merged_df = df[df["category"] == group_cats[0]].copy()
                representative_cat = group_cats[0]

            # Eğit — representative_cat profil parametrelerini belirler
            try:
                success = self.train_category(representative_cat, merged_df, verbose=verbose)
                trained_keys.add(model_key)
                for cat in group_cats:
                    results[cat] = success
            except Exception as e:
                logger.error(f"❌ [{model_key}] Eğitim hatası: {e}")
                for cat in group_cats:
                    results[cat] = False

        trained = sum(1 for v in results.values() if v)
        logger.info(f"📊 Eğitim sonucu: {trained}/{len(categories)} kategori ({len(trained_keys)} model)")
        return results

    def save_all(self):
        """Tüm eğitilmiş modelleri diske kaydeder."""
        for cat, predictor in self._models.items():
            if predictor.is_trained:
                try:
                    predictor.save(version=1)
                except Exception as e:
                    logger.error(f"Model kaydetme hatası ({cat}): {e}")

    def load_all(self):
        """
        models/ dizinindeki tüm alt klasörlerden modelleri yükler.
        Her alt klasör bir kategori adı olarak ele alınır.
        """
        if not os.path.exists(self._model_base_dir):
            return

        for entry in os.listdir(self._model_base_dir):
            cat_dir = os.path.join(self._model_base_dir, entry)
            model_file = os.path.join(cat_dir, "catboost_v1.cbm")
            if os.path.isdir(cat_dir) and os.path.exists(model_file):
                self.get_or_create(entry)  # Otomatik yükleme yapar

    # ─── Status / Health ──────────────────────────────────────────────────────

    @property
    def categories(self) -> list[str]:
        """Kayıtlı kategori listesi."""
        return list(self._models.keys())

    @property
    def trained_categories(self) -> list[str]:
        """Eğitilmiş kategori listesi."""
        return [c for c, m in self._models.items() if m.is_trained]

    def status(self) -> dict:
        """Tüm model durumu özeti."""
        return {
            "total_categories": len(self._models),
            "trained_categories": len(self.trained_categories),
            "categories": {
                cat: {
                    "trained": predictor.is_trained,
                    "profile": self.get_profile(cat).get("profile_type", "unknown"),
                    "metrics": predictor.metrics if predictor.is_trained else {},
                    "train_stats": self._train_stats.get(cat, {}),
                }
                for cat, predictor in self._models.items()
            },
        }
