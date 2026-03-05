# algorithms/catboost_model.py
"""
Algoritma 2: CatBoost (Categorical Boosting) — v2
Ana talep tahmini — 15+ JSONB attribute + tüm mühendislik feature'ları.

v2 İyileştirmeler:
- 15 JSONB attribute kategorik feature olarak
- Time-based train/test split
- SHAP feature açıklanabilirlik
- Accuracy metrikleri (MAE, MAPE, R²)
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from config import (
    CATBOOST_ITERATIONS, CATBOOST_LEARNING_RATE,
    CATBOOST_DEPTH, CATBOOST_FORECAST_DAYS
)


class DemandPredictor:
    """
    CatBoost v2 — 40+ feature ile talep tahmini.

    Kategorik alanlar: brand, fabric, color, pattern, category + 15 JSONB attr
    Sayısal: rolling avg, momentum, velocity, rank change, size depletion...
    """

    # Statik kategorik feature'lar
    BASE_CAT_FEATURES = ["category", "fabric", "color", "pattern", "brand"]

    # JSONB'den gelen kategorik feature'lar
    JSONB_CAT_FEATURES = [
        "attr_sezon", "attr_koleksiyon", "attr_boy", "attr_kalıp",
        "attr_siluet", "attr_yaka_tipi", "attr_kol_boyu", "attr_kol_tipi",
        "attr_bel", "attr_paça_tipi", "attr_desen", "attr_dokuma_tipi",
        "attr_persona", "attr_materyal_bileşeni", "attr_astar_durumu",
    ]

    # Diğer string feature'lar (FeatureEngineer'dan)
    OTHER_CAT_FEATURES = [
        "anomaly_flag", "current_regime", "discount_intensity",
    ]

    # Tüm sayısal feature adayları
    NUMERIC_FEATURES = [
        "price", "discount_rate", "cart_count", "favorite_count",
        "view_count", "rating", "rating_count", "engagement_score",
        # v2: yeni numerikler
        "rank_change_3d", "abs_rank_change_3d", "rank_improving",
        "size_depletion_rate", "rapid_size_depletion", "size_change_velocity",
        "product_age_days", "is_new_product",
        "day_of_week", "is_weekend", "week_of_year", "month",
        "discount_change", "discount_increasing",
        "stock_flip_count", "is_out_of_stock",
        "price_vs_category_avg", "engagement_vs_category", "cart_vs_category",
        "days_since_changepoint", "cluster_id", "visual_trend_score",
    ]

    TARGET = "target_demand"

    def __init__(self, model_dir="models"):
        self.model = None
        self.model_dir = model_dir
        self.feature_importance = {}
        self.is_trained = False
        self.metrics = {}
        self.shap_values = None
        self._feature_names = []
        os.makedirs(model_dir, exist_ok=True)

    def _collect_features(self, df: pd.DataFrame) -> tuple:
        """Mevcut sütunlardan feature listesi oluşturur."""
        # Sayısal
        numeric = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        # Rolling, momentum, velocity (dinamik)
        for col in df.columns:
            if col.startswith("rolling_") or col.startswith("momentum_") or col.endswith("_velocity"):
                if col not in numeric:
                    numeric.append(col)
            elif col.startswith("z_") and col not in numeric:
                numeric.append(col)

        # Kategorik
        cat = [c for c in self.BASE_CAT_FEATURES if c in df.columns]
        cat += [c for c in self.JSONB_CAT_FEATURES if c in df.columns]
        cat += [c for c in self.OTHER_CAT_FEATURES if c in df.columns]

        # Deduplicate
        seen = set()
        all_features = []
        for f in numeric + cat:
            if f not in seen:
                all_features.append(f)
                seen.add(f)

        cat_indices = [all_features.index(c) for c in cat if c in all_features]

        return all_features, cat, cat_indices

    @staticmethod
    def _compute_composite_target(recent: pd.DataFrame) -> float:
        """
        [ADIM 2] Bileşik talep skoru — DEĞIŞIM bazlı.

        ESKİ SORUN:
          log(favorite_count) mutlak birikimi ölçüyor.
          Sezonsal düşen ürün (geçmişte çok satmış) hâlâ yüksek favori → yanlış yüksek skor.

        YENİ YAKLAŞIM:
          favorite_growth_14d = son 7 gün ort / önceki 7 gün ort
          > 1.0 = artıyor (rising sinyal)
          < 1.0 = azalıyor (falling sinyal)
          Geçmiş birikimden bağımsız → sezonsal asimetri ortadan kalkıyor.

        Ağırlıklar:
          rank_imp       %40 → Gecikmesiz, erken sinyal
          fav_growth     %35 → Değişim bazlı, geçmiş bağımsız
          rating_count   %15 → Gecikmeli ama gerçek satış kanıtı
          cart_count     %10 → Bonus (nadiren dolu)
          price_penalty  ceza → Hızlı fiyat artışı talebi baskılar
        """
        rate  = recent["rating_count"].fillna(0).mean()
        cart  = recent["cart_count"].fillna(0).mean()

        # Favori BÜYÜME ORANI (mutlak değil)
        if "favorite_growth_14d" in recent.columns:
            fav_growth = float(recent["favorite_growth_14d"].fillna(1.0).mean())
        else:
            # Fallback: son / ilk ratio
            fav_vals = recent["favorite_count"].fillna(0).values
            if len(fav_vals) >= 2 and fav_vals[0] > 0:
                fav_growth = float(fav_vals[-1]) / float(fav_vals[0])
            else:
                fav_growth = 1.0
        fav_growth = max(0.05, min(10.0, fav_growth))

        # Rank improvement
        if "absolute_rank" in recent.columns and len(recent) >= 2:
            ranks = recent["absolute_rank"].dropna().values
            if len(ranks) >= 2:
                rank_raw = float(ranks[0] - ranks[-1])
                rank_imp = max(-1.0, min(1.0, rank_raw / 1000.0))
            else:
                rank_imp = 0.0
            if "rank_momentum_score" in recent.columns:
                mom = recent["rank_momentum_score"].fillna(0).mean()
                rank_imp = rank_imp * 0.5 + float(mom) * 0.5
        else:
            rank_imp = 0.0

        # Fiyat baskısı cezası
        if "price_change_pct" in recent.columns:
            avg_price_change = float(recent["price_change_pct"].fillna(0).mean())
            if avg_price_change > 20:
                price_penalty = 0.80   # %20+ artış → %20 ceza
            elif avg_price_change > 10:
                price_penalty = 0.92   # %10-20 artış → %8 ceza
            else:
                price_penalty = 1.0
        else:
            price_penalty = 1.0

        # ── Hesaplama ──────────────────────────────────────────────
        # NOT: fav_growth > 1 → log > 0, fav_growth < 1 → log < 0
        # Bu negatif değer DUSEN sinyali verir
        score = (
            np.clip(rank_imp, -1, 1)    * 0.40 +
            np.log(fav_growth)          * 0.35 +   # değişim oranı (negatif olabilir)
            np.log1p(rate)              * 0.15 +
            np.log1p(cart)              * 0.10
        ) * price_penalty

        return max(0.0, round(float(score), 4))


    def prepare_training_data(self, df: pd.DataFrame, forecast_days=None) -> tuple:
        """
        DailyMetric → eğitim seti.
        Her ürünün son gün feature'ları → composite_demand_score target.

        Hedef değişken neden değişti?
          Eski: cart_count.mean()  → %91 sıfır, CatBoost hep 0 tahmin ediyor
          Yeni: composite_demand_score → favorite + rating + rank + (cart varsa)
        """
        if forecast_days is None:
            forecast_days = CATBOOST_FORECAST_DAYS

        all_features, cat_features, cat_indices = self._collect_features(df)

        rows = []
        for pid in df["product_id"].unique():
            product_data = df[df["product_id"] == pid].sort_values("date")

            if len(product_data) < 7:
                continue

            last_row = product_data.iloc[-1]
            recent = product_data.tail(7)

            # ── YENİ: Bileşik hedef değişken ──────────────────────
            target = self._compute_composite_target(recent)

            row = {}
            for col in all_features:
                val = last_row.get(col, None)
                if col in [c for c in cat_features]:
                    row[col] = str(val) if val is not None else "unknown"
                else:
                    try:
                        row[col] = float(val) if val is not None and val == val else 0.0
                    except (ValueError, TypeError):
                        row[col] = 0.0

            row[self.TARGET] = target
            row["product_id"] = pid
            rows.append(row)

        if not rows:
            return None, None, None, None

        train_df = pd.DataFrame(rows)

        # NaN doldur
        for col in all_features:
            if col in cat_features:
                train_df[col] = train_df[col].fillna("unknown").astype(str)
            else:
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)

        X = train_df[all_features]
        y = train_df[self.TARGET]

        # Recompute cat_indices from actual X columns
        all_cats = set(self.BASE_CAT_FEATURES + self.JSONB_CAT_FEATURES + self.OTHER_CAT_FEATURES)
        cat_indices = [i for i, col in enumerate(X.columns) if col in all_cats]

        return X, y, cat_indices, all_features

    def train(self, df: pd.DataFrame, verbose=True):
        """
        v2 Eğitim: Time-based split + accuracy metrikleri.
        """
        X, y, cat_indices, features = self.prepare_training_data(df)

        if X is None or len(X) < 5:
            print("  ⚠ CatBoost: Yeterli eğitim verisi yok")
            return False

        self._feature_names = features

        # ── TIME-BASED SPLIT ────────────────────────────────────
        # Son %20'yi test olarak ayır (zamana göre, random değil)
        split = int(len(X) * 0.8)
        if split < 3:
            split = len(X)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self.model = CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS,
            learning_rate=CATBOOST_LEARNING_RATE,
            depth=CATBOOST_DEPTH,
            cat_features=cat_indices,
            verbose=0,
            random_seed=42,
            early_stopping_rounds=50,
            l2_leaf_reg=3,
            border_count=128,
        )

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        eval_pool = Pool(X_test, y_test, cat_features=cat_indices) if len(X_test) > 0 else None

        self.model.fit(train_pool, eval_set=eval_pool, verbose=0)

        # ── FEATURE IMPORTANCE ──────────────────────────────────
        importances = self.model.get_feature_importance()
        self.feature_importance = {
            features[i]: round(importances[i], 2)
            for i in range(len(features))
        }
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        # ── ACCURACY METRİKLERİ ── [YENİ] ──────────────────────
        self.metrics = {}
        if eval_pool and len(X_test) > 0:
            y_pred = self.model.predict(X_test)
            self.metrics = self._calculate_metrics(y_test.values, y_pred)

        self.is_trained = True

        # ── SHAP ── [YENİ] ─────────────────────────────────────
        try:
            self.shap_values = self.model.get_feature_importance(
                Pool(X_train, cat_features=cat_indices),
                type="ShapValues"
            )
        except Exception:
            self.shap_values = None

        if verbose:
            cat_count = len([c for c in features if c in
                           self.BASE_CAT_FEATURES + self.JSONB_CAT_FEATURES + self.OTHER_CAT_FEATURES])
            num_count = len(features) - cat_count
            print(f"  ✓ CatBoost eğitildi ({len(X)} örnek, {len(features)} feature)")
            print(f"    → {num_count} sayısal + {cat_count} kategorik")

            top5 = list(self.feature_importance.items())[:5]
            print(f"    Top-5 feature:")
            for i, (k, v) in enumerate(top5):
                bar = "█" * int(v / 2)
                print(f"      {i+1}. {k}: {v}% {bar}")

            if self.metrics:
                print(f"    📊 Doğruluk:")
                print(f"       MAE: {self.metrics['mae']:.2f}")
                print(f"       MAPE: {self.metrics['mape']:.1f}%")
                print(f"       R²: {self.metrics['r2']:.3f}")

        return True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tahmin yapar — egitimdeki feature sirasini kullanir."""
        if not self.is_trained or not self._feature_names:
            print("  CatBoost henuz egitilmedi!")
            return pd.DataFrame()

        # Egitim sirasindaki feature'lari kullanarak X olustur
        all_cats = set(self.BASE_CAT_FEATURES + self.JSONB_CAT_FEATURES + self.OTHER_CAT_FEATURES)
        rows = []
        for pid in df["product_id"].unique():
            pdata = df[df["product_id"] == pid].sort_values("date")
            if len(pdata) < 7:
                continue
            last_row = pdata.iloc[-1]
            row = {}
            for col in self._feature_names:
                val = last_row.get(col, None)
                if col in all_cats:
                    row[col] = str(val) if val is not None else "unknown"
                else:
                    try:
                        row[col] = float(val) if val is not None and val == val else 0.0
                    except (ValueError, TypeError):
                        row[col] = 0.0
            row["product_id"] = pid
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        pred_df = pd.DataFrame(rows)
        X = pred_df[self._feature_names]

        # NaN doldur
        for col in self._feature_names:
            if col in all_cats:
                X[col] = X[col].fillna("unknown").astype(str)
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        predictions = self.model.predict(X)

        result = pd.DataFrame({
            "product_id": pred_df["product_id"].values,
            "predicted_demand": np.round(np.maximum(predictions, 0)).astype(int),
        })

        return result

    def predict_for_features(self, features_dict: dict) -> float:
        """Tek feature set için tahmin."""
        if not self.is_trained:
            return 0.0

        df = pd.DataFrame([features_dict])
        all_cats = self.BASE_CAT_FEATURES + self.JSONB_CAT_FEATURES + self.OTHER_CAT_FEATURES
        for col in all_cats:
            if col in df.columns:
                df[col] = df[col].astype(str)

        try:
            pred = self.model.predict(df)
            return max(0.0, float(pred[0]))
        except Exception:
            return 0.0

    def predict_for_representative(self, typical_row: pd.Series,
                                    override: dict = None) -> float:
        """
        Gerçek eğitim satırını kullanarak tahmin yapar.
        Sadece belirtilen kolonlar override edilir (örn. fabric, color).

        Bu sayede rolling_avg_*, momentum_* gibi tüm feature'lar
        gerçek veriden gelir — model sıfır tahmin yapmaz.
        """
        if not self.is_trained or not self._feature_names:
            return 0.0

        all_cats = set(self.BASE_CAT_FEATURES + self.JSONB_CAT_FEATURES +
                       self.OTHER_CAT_FEATURES)
        row = {}
        for col in self._feature_names:
            # Override varsa kullan
            if override and col in override:
                row[col] = str(override[col]) if col in all_cats else override[col]
            elif col in typical_row.index:
                val = typical_row[col]
                if col in all_cats:
                    row[col] = str(val) if pd.notna(val) else "unknown"
                else:
                    try:
                        row[col] = float(val) if pd.notna(val) else 0.0
                    except (ValueError, TypeError):
                        row[col] = 0.0
            else:
                row[col] = "unknown" if col in all_cats else 0.0

        df = pd.DataFrame([row])
        for col in self._feature_names:
            if col in all_cats:
                df[col] = df[col].fillna("unknown").astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        try:
            pred = self.model.predict(df[self._feature_names])
            return max(0.0, float(pred[0]))
        except Exception:
            return 0.0

    def get_shap_explanation(self, top_n=10) -> dict:
        """
        [YENİ] SHAP feature açıklanabilirlik raporu.
        """
        if self.shap_values is None:
            return {"error": "SHAP değerleri hesaplanmadı"}

        mean_abs_shap = np.abs(self.shap_values[:, :-1]).mean(axis=0)
        feature_shap = {
            self._feature_names[i]: round(float(mean_abs_shap[i]), 4)
            for i in range(len(self._feature_names))
        }
        feature_shap = dict(sorted(feature_shap.items(), key=lambda x: x[1], reverse=True))

        return {
            "top_features": dict(list(feature_shap.items())[:top_n]),
            "total_features": len(feature_shap),
            "explanation": "SHAP değeri yüksek = o feature tahmin sonucunu daha çok etkiliyor",
        }

    def _calculate_metrics(self, y_true, y_pred) -> dict:
        """
        [YENİ] MAE, MAPE, R² hesapla.
        """
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-6))

        return {
            "mae": round(float(mae), 2),
            "mape": round(float(mape), 1),
            "r2": round(float(r2), 4),
            "samples": len(y_true),
        }

    def save(self, version=1):
        """Model dosyasını kaydeder."""
        path = os.path.join(self.model_dir, f"catboost_v{version}.cbm")
        if self.model:
            self.model.save_model(path)
            print(f"  ✓ Model kaydedildi: {path}")

    def load(self, version=1):
        """Model dosyasını yükler."""
        path = os.path.join(self.model_dir, f"catboost_v{version}.cbm")
        if os.path.exists(path):
            self.model = CatBoostRegressor()
            self.model.load_model(path)
            self.is_trained = True
            print(f"  ✓ Model yüklendi: {path}")
